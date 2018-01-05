# -*- coding: utf-8 -*-

import cv2
import numpy as np
from scipy.ndimage import generic_gradient_magnitude, sobel

POS_MASK = 100000
NEG_MASK = -100


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def generate_mask(start_x, start_y, end_x, end_y, img, is_pos):
    width, height = img.size
    mask = np.ones((height, width))

    if start_x > end_x:
        start_x, end_x = end_x, start_x

    if start_y > end_y:
        start_y, end_y = end_y, start_y

    if is_pos:
        mask[start_y:(end_y + 1), start_x:(end_x + 1)] *= POS_MASK
    else:
        mask[start_y:(end_y + 1), start_x:(end_x + 1)] *= NEG_MASK

    return mask


def rotate_image(image, ccw):
    """
    Rotate numpy array (image) by 90 degrees
    :param image: image to rotate
    :param ccw: flag to rotate counter-clock wise.
    :return: image
    """
    height, width, ch = image.shape
    output = np.zeros((width, height, ch))
    if ccw:
        image_flip = np.fliplr(image)
        for c in range(ch):
            for row in range(height):
                output[:, row, c] = image_flip[row, :, c]
    else:
        for c in range(ch):
            for row in range(height):
                output[:, height - 1 - row, c] = image[row, :, c]
    return output


def find_horizontal_seam(energy_matrix):
    """
    Takes a img and returns the lowest energy vertical seam as a list of pixels (2-tuples).
    This implements the dynamic programming seam-find algorithm. For an m*n picture, this algorithm
    takes O(m*n) time
    """
    cost = calc_cost_matrix(energy_matrix)
    path = find_seam(cost, energy_matrix)

    return path


def find_seam(cost, energy_matrix):
    """
    Calculate optimal seam using cost_matrix and original energy matrix
    :param cost: cost matrix
    :param energy_matrix: energy matrix calculated by gradient filter (Sobel)
    :return: optimal path to traverse the image.
    """
    min_val = None
    start_point = None
    path = []
    im_height, im_width = energy_matrix.shape

    for y in range(im_width):
        if not min_val or cost[im_height - 1, y] < min_val:
            min_val = cost[im_height - 1, y]
            start_point = y
    pos = (im_height - 1, start_point)
    path.append(pos)
    while pos[0] != 0:
        val = cost[pos] - energy_matrix[pos]
        x, y = pos
        if y == 0:
            if val == cost[x - 1, y + 1]:
                pos = (x - 1, y + 1)
            else:
                pos = (x - 1, y)
        elif y < im_width - 2:
            if val == cost[x - 1, y + 1]:
                pos = (x - 1, y + 1)
            elif val == cost[x - 1, y]:
                pos = (x - 1, y)
            else:
                pos = (x - 1, y - 1)
        else:
            if val == cost[x - 1, y]:
                pos = (x - 1, y)
            else:
                pos = (x - 1, y - 1)

        path.append(pos)
    return path


def calc_cost_matrix(energy_matrix):
    """
    Calculate cumulative cost matrix
    :param energy_matrix: energy matrix calculated by gradient filter (Sobel)
    :return: cumulative cost matrix
    """
    im_height, im_width = energy_matrix.shape
    cost = np.zeros((im_height, im_width))
    for y in range(im_width):
        cost[0, y] = energy_matrix[0, y]
    for x in range(1, im_height):
        for y in range(im_width):
            if y == 0:
                min_val = min(cost[x - 1, y], cost[x - 1, y + 1])
            elif y < im_width - 2:
                min_val = min(cost[x - 1, y], cost[x - 1, y + 1])
                min_val = min(min_val, cost[x - 1, y - 1])
            else:
                min_val = min(cost[x - 1, y], cost[x - 1, y - 1])
            cost[x, y] = energy_matrix[x, y] + min_val
    return cost


class SeamCarver:
    def __init__(self, file_path, out_height, out_width, mask=None):
        # initialize parameter
        self.file_path = file_path
        self.out_height = out_height
        self.out_width = out_width
        self.mask = mask

        # read in image and store as np.float64 format
        self.in_image = cv2.imread(file_path).astype(np.float64)
        self.in_height, self.in_width = self.in_image.shape[: 2]

        # keep tracking resulting image
        self.out_image = np.copy(self.in_image)

        # kernel for forward energy map calculation
        self.kernel_x = np.array([[0., 0., 0.], [-1., 0., 1.], [0., 0., 0.]], dtype=np.float64)
        self.kernel_y_left = np.array([[0., 0., 0.], [0., 0., 1.], [0., -1., 0.]], dtype=np.float64)
        self.kernel_y_right = np.array([[0., 0., 0.], [1., 0., 0.], [0., -1., 0.]], dtype=np.float64)

        # starting program
        self.seams_carving()

    def seams_carving(self):
        """
        Process image vertically by adding or removing number of pixels iteratively, then rotate
        the image 90 degrees and di the same for horizontal.
        """

        # calculate number of rows and columns needed to be inserted or removed
        d_height, d_width = int(self.out_height - self.in_height), int(self.out_width - self.in_width)

        # remove column
        if d_width < 0:
            self.seams_removal(np.abs(d_width))
        # insert column
        elif d_width > 0:
            self.seams_insertion(d_width)

        # remove row
        if d_height < 0:
            self.out_image = rotate_image(self.out_image, 1)
            self.seams_removal(np.abs(d_height))
            self.out_image = rotate_image(self.out_image, 0)
        # insert row
        elif d_height > 0:
            self.out_image = rotate_image(self.out_image, 1)
            self.seams_insertion(d_height)
            self.out_image = rotate_image(self.out_image, 0)

    def seams_removal(self, num_pixel):
        """
        Calculate minimal energy seams and remove those seams from the image
        :param num_pixel: number of pixels (width) to remove
        """
        for p in range(num_pixel):
            energy_matrix = self.gradient_filter()
            seam = find_horizontal_seam(energy_matrix)
            output = self.delete_seam(seam)
            self.out_image = output

    def seams_insertion(self, num_pixel):
        """
        Calculate minimal energy seams and create new similar seams to enlarge the image
        :param num_pixel: number of pixel columns to add
        """

        for dummy in range(num_pixel):
            energy_matrix = self.gradient_filter()
            seam = find_horizontal_seam(energy_matrix)
            new_image = self.add_vertical_seam(seam)
            self.out_image = new_image

    def add_vertical_seam(self, path):
        """
        Adds the pixels in a vertical path from img
        @img: an input img
        @path: pixels to delete in a vertical path
        """

        img_height, img_width, dim = self.out_image.shape

        output = np.zeros((img_height, img_width + 1, dim))
        path_set = set(path)
        seen_set = set()
        for x in range(img_height):
            for y in range(img_width):
                if (x, y) not in path_set and x not in seen_set:
                    output[x, y] = self.out_image[x, y]
                elif (x, y) in path_set and x not in seen_set:
                    output[x, y] = self.out_image[x, y]
                    seen_set.add(x)
                    if y < img_width - 2:
                        avg = np.mean([self.out_image[x, y], self.out_image[x, y + 1]], 0)
                        output[x, y + 1] = avg
                    else:
                        output[x, y + 1] = np.mean([self.out_image[x, y], self.out_image[x, y - 1]], 0)
                else:
                    y_ = self.out_image[x, y]
                    output[x, y + 1] = y_

        return output

    def gradient_filter(self):

        """
        Takes a grayscale img and retuns the Sobel operator on the image. Fast thanks to Scipy/Numpy.
        """
        im_height, im_width, _ = self.out_image.shape
        bw_image = rgb2gray(self.out_image)
        sobel_arr = generic_gradient_magnitude(bw_image, derivative=sobel)
        # gradient_sum = np.sum(sobel_arr, axis=1)
        return sobel_arr

    def delete_seam(self, seam_idx):
        height, width, dim = self.out_image.shape
        output = np.zeros((height, width - 1, dim))
        for h, w in seam_idx:
            output[h] = np.delete(self.out_image[h], w, 0)
        return output

    def save_result(self, filename):
        """
        Save output file
        :param filename: filename to use
        """
        cv2.imwrite(filename, self.out_image.astype(np.uint8))
