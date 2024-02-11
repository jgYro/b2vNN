import argparse
import math
from PIL import Image


def create_coordinates(file_path):
    coordinates = []
    with open(file_path, 'rb') as file:
        window = file.read(2)
        while len(window) == 2:
            coordinates.append((window[0], window[1]))
            window = file.read(2)
    return coordinates


def create_image(coordinates, image_size):
    image = Image.new('L', image_size)  # 'L' mode for greyscale
    pixel_values = [0] * (image_size[0] * image_size[1])

    for coord in coordinates:
        index = coord[1] * image_size[0] + coord[0]
        pixel_values[index] += 1

    max_frequency = max(pixel_values)
    if max_frequency == 0:
        max_frequency = 1  # Avoid division by zero if there are no coordinates

    for y in range(image_size[1]):
        for x in range(image_size[0]):
            index = y * image_size[0] + x
            brightness = int((math.log(pixel_values[index] + 1) / math.log(max_frequency + 1)) * 255)
            image.putpixel((x, y), brightness)

    return image


def main():
    parser = argparse.ArgumentParser(description="Generate an image based on the frequency of coordinates from a binary file.")
    parser.add_argument("file_path", help="Path to the binary file")
    parser.add_argument("--image-size", nargs=2, type=int, default=[256, 256], metavar=('WIDTH', 'HEIGHT'),
                        help="Size of the output image (default: 256 256)")
    parser.add_argument("--output", "-o", help="Output file path to save the generated PNG image")
    args = parser.parse_args()

    coordinates = create_coordinates(args.file_path)
    image = create_image(coordinates, tuple(args.image_size))
    if args.output:
        image.save(args.output)
        print(f"Image saved to {args.output}")
    else:
        image.show()


if __name__ == "__main__":
    main()
