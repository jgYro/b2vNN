from PIL import Image
import os

def convert_images_to_greyscale(directory):
    """
    Convert all PNG images in the specified directory to greyscale.
    """
    # Check if the directory exists
    if not os.path.exists(directory):
        print(f"The directory {directory} does not exist.")
        return

    # Loop through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            # Construct the full file path
            filepath = os.path.join(directory, filename)
            try:
                # Open the image
                with Image.open(filepath) as img:
                    # Convert the image to greyscale
                    grey_img = img.convert("L")
                    # Save the greyscale image back to disk
                    grey_img.save(filepath)
                    print(f"Converted {filename} to greyscale.")
            except Exception as e:
                print(f"Failed to convert {filename} to greyscale. Error: {e}")


# Usage example
directory = "./test/1/"
convert_images_to_greyscale(directory)
directory = "./train/1/"
convert_images_to_greyscale(directory)
