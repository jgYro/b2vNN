import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image


def predict_image(model, img_path):
    # Load the image
    img = image.load_img(img_path, target_size=(128, 128), color_mode='grayscale')
    # Convert the image to a numpy array
    img_array = image.img_to_array(img)
    # Expand the dimensions of the image to match the input shape of your model
    # (add batch dimension)
    img_array = np.expand_dims(img_array, axis=0)
    # Preprocess the image (same preprocessing as training)
    img_array = img_array / 255.0  # rescale to [0, 1]
    # Make predictions
    predictions = model.predict(img_array)
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Image Prediction")
    parser.add_argument("image_path", help="Path to the image")
    parser.add_argument("model_path", help="Path to the model")
    args = parser.parse_args()

    # Load the model
    model = tf.keras.models.load_model(args.model_path)

    # Make predictions
    predictions = predict_image(model, args.image_path)
    print(predictions)


if __name__ == "__main__":
    main()
