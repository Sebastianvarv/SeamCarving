import argparse

from seam_carver import SeamCarver


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Seam carve image resizing')
    parser.add_argument("-i", "--image_path", help="Image path", required=True)
    parser.add_argument("-o", "--out_name", help="Output filename", required=False, default="output.png")
    parser.add_argument("-d", "--height", help="output height", required=True, type=int)
    parser.add_argument("-w", "--width", help="output width", required=True, type=int)

    args = parser.parse_args()

    obj = SeamCarver(args.image_path, args.height, args.width)
    obj.save_result(args.out_name)
