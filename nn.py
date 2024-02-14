import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define data generators
train_datagenerator = ImageDataGenerator(rescale=1./255)
test_datagenerator = ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 40 using train_datagenerator
train_generator = train_datagenerator.flow_from_directory(
    'train',
    target_size=(128, 128),
    batch_size=40,
    class_mode='binary',
    color_mode='grayscale'  # Specify color mode as grayscale
)

# Flow validation images in batches of 40 using test_datagenerator
validation_generator = test_datagenerator.flow_from_directory(
    'test',
    target_size=(128, 128),
    batch_size=40,
    class_mode='binary',
    color_mode='grayscale'  # Specify color mode as grayscale
)

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                           input_shape=(128, 128, 1)),  # Input shape with channel dimension
    tf.keras.layers.MaxPooling2D((2, 2), 2),

    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2), 2),

    tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2), 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(0.001),
              metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    epochs=25,
    validation_data=validation_generator,
)


model.save('test.h5')
