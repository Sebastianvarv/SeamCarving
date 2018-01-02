import numpy as np
import cv2


def filter_output(b, g, r, kernel):
    output = np.absolute(cv2.filter2D(b, -1, kernel=kernel)) + \
             np.absolute(cv2.filter2D(g, -1, kernel=kernel)) + \
             np.absolute(cv2.filter2D(r, -1, kernel=kernel))
    return output


def cumulative_map_backward(energy_map):
    height, width = energy_map.shape
    output = np.copy(energy_map)
    for row in range(1, height):
        for col in range(width):
            output[row, col] = \
                energy_map[row, col] + np.amin(output[row - 1, max(col - 1, 0): min(col + 2, width - 1)])
    return output


def find_seam(cumulative_map):
    """
    Find minimal cost seam using cumulative map using greedy algorithm
    :param cumulative_map: Cumulative cost matrix
    :return: Seam which is list of pixels
    """
    height, width = cumulative_map.shape
    output = np.zeros((height,), dtype=np.uint32)
    output[-1] = np.argmin(cumulative_map[-1])
    for row in range(height - 2, -1, -1):
        prv_x = output[row + 1]
        if prv_x == 0:
            output[row] = np.argmin(cumulative_map[row, : 2])
        else:
            output[row] = np.argmin(cumulative_map[row, prv_x - 1: min(prv_x + 2, width - 1)]) + prv_x - 1
    return output


def update_seams(remaining_seams, current_seam):
    """
    Upcate seams after adding a new one
    :param remaining_seams: remaining seams to traverse
    :param current_seam: seam which was just added
    :return: list of seams to traverse
    """
    output = []
    for seam in remaining_seams:
        seam[np.where(seam >= current_seam)] += 2
        output.append(seam)
    return output


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


class SeamCarver:
    def __init__(self, file_path, out_height, out_width):
        # initialize parameter
        self.file_path = file_path
        self.out_height = out_height
        self.out_width = out_width

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
            energy_matrix = self.calc_energy_map()
            cost_matrix = self.cumulative_map_forward(energy_matrix)
            seam_idx = find_seam(cost_matrix)
            self.delete_seam(seam_idx)

    def seams_insertion(self, num_pixel):
        """
        Calculate minimal energy seams and create new similar seams to enlarge the image
        :param num_pixel: number of pixel columns to add
        """

        temp_image = np.copy(self.out_image)
        seams_record = []

        for dummy in range(num_pixel):
            energy_map = self.calc_energy_map()
            cumulative_map = cumulative_map_backward(energy_map)
            seam_idx = find_seam(cumulative_map)
            seams_record.append(seam_idx)
            self.delete_seam(seam_idx)

        self.out_image = np.copy(temp_image)
        n = len(seams_record)
        for dummy in range(n):
            seam = seams_record.pop(0)
            self.add_seam(seam)
            seams_record = update_seams(seams_record, seam)

    def calc_energy_map(self):
        """
        Calculate energy map for the whole image for each color channel.  The energy of pixel (x, y) is
        Δx2(x, y) + Δy2(x, y), where the square of the x-gradient Δx2(x, y) = Rx(x, y)2 + Gx(x, y)2 + Bx(x, y)2,
        and where the central differences Rx(x, y), Gx(x, y), and Bx(x, y) are the absolute value in differences of
        red, green, and blue components between pixel (x + 1, y) and pixel (x − 1, y).
        :return: Matrix m x n where m is height of the image and n is width. Each value at (x,y) corresponds to pixel
        energy value.
        """
        b, g, r = cv2.split(self.out_image)
        b_energy = np.absolute(cv2.Scharr(b, -1, 1, 0)) + np.absolute(cv2.Scharr(b, -1, 0, 1))
        g_energy = np.absolute(cv2.Scharr(g, -1, 1, 0)) + np.absolute(cv2.Scharr(g, -1, 0, 1))
        r_energy = np.absolute(cv2.Scharr(r, -1, 1, 0)) + np.absolute(cv2.Scharr(r, -1, 0, 1))
        return b_energy + g_energy + r_energy

    def cumulative_map_forward(self, energy_map):
        """
        Build accumulative cost matrix using dynamic programming
        :param energy_map: Previously calculated energy map where each pixel is assigned energy
        :return: Cumulative cost matrix.
        """
        matrix_x, matrix_y_left, matrix_y_right = self.calc_neighbor_matrix()

        height, width = energy_map.shape
        output = np.copy(energy_map)
        for row in range(1, height):
            for col in range(width):
                e_right = output[row - 1, col + 1] + matrix_x[row - 1, col + 1] + matrix_y_right[row - 1, col + 1]
                e_up = output[row - 1, col] + matrix_x[row - 1, col]
                e_left = output[row - 1, col - 1] + matrix_x[row - 1, col - 1] + matrix_y_left[row - 1, col - 1]

                if col == 0:
                    output[row, col] = energy_map[row, col] + min(e_right, e_up)
                elif col == width - 1:
                    output[row, col] = energy_map[row, col] + min(e_left, e_up)
                else:
                    output[row, col] = energy_map[row, col] + min(e_left, e_right, e_up)
        return output

    def calc_neighbor_matrix(self):
        """
        Calculate neighbouring matrix for each color channel.
        :return: Neighbouring matrices
        """
        b, g, r = cv2.split(self.out_image)
        mat_x = filter_output(b, g, r, self.kernel_x)
        mat_y_l = filter_output(b, g, r, self.kernel_y_left)
        mar_y_r = filter_output(b, g, r, self.kernel_y_right)
        return mat_x, mat_y_l, mar_y_r

    def delete_seam(self, seam_idx):
        """
        Delete seam from the image in order to reduce image dimensionality
        :param seam_idx: List of pixels containing a seam to remove
        """
        m, n = self.out_image.shape[: 2]
        output = np.zeros((m, n - 1, 3))
        for row in range(m):
            col = seam_idx[row]
            output[row, :, 0] = np.delete(self.out_image[row, :, 0], [col])
            output[row, :, 1] = np.delete(self.out_image[row, :, 1], [col])
            output[row, :, 2] = np.delete(self.out_image[row, :, 2], [col])
        self.out_image = np.copy(output)

    def add_seam(self, seam_idx):
        """
        Add seam to enlarge the image
        :param seam_idx: List of pixels containing a seam to duplicate
        :return:
        """
        m, n = self.out_image.shape[: 2]
        output = np.zeros((m, n + 1, 3))
        for row in range(m):
            col = seam_idx[row]
            for ch in range(3):
                if col == 0:
                    p = np.average(self.out_image[row, col: col + 2, ch])
                    output[row, col, ch] = self.out_image[row, col, ch]
                    output[row, col + 1, ch] = p
                    output[row, col + 1:, ch] = self.out_image[row, col:, ch]
                else:
                    p = np.average(self.out_image[row, col - 1: col + 1, ch])
                    output[row, : col, ch] = self.out_image[row, : col, ch]
                    output[row, col, ch] = p
                    output[row, col + 1:, ch] = self.out_image[row, col:, ch]
        self.out_image = np.copy(output)

    def save_result(self, filename):
        """
        Save output file
        :param filename: filename to use
        """
        cv2.imwrite(filename, self.out_image.astype(np.uint8))
