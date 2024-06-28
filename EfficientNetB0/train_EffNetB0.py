import os
import numpy as np
import tensorflow as tf
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.applications import EfficientNetB0
from keras._tf_keras.keras.layers import Dense, GlobalAveragePooling2D
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.callbacks import ModelCheckpoint

# Set up paths to the dataset
train_dir = './Dataset'

# Define image dimensions and batch size
img_height, img_width = 224, 224
batch_size = 16

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # Splitting 80% for training, 20% for validation
)

# Generate batches of augmented data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',  # since we have two classes: empty and not_empty
    subset='training'  # specify this is for training set
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'  # specify this is for validation set
)

# Load EfficientNetB0 base model without the top layer
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Add a global average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Add a fully-connected layer
x = Dense(1, activation='relu')(x)

# Add output layer with sigmoid activation for binary classification
predictions = Dense(1, activation='sigmoid')(x)

# Combine the base model and custom layers
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base model layers (optional)
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Define callbacks - including ModelCheckpoint to save best weights
checkpoint_filepath = './Project Car Detection EfficientNet/EffNetB0checkpoint.weights.h5'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True
)

# Calculate number of steps per epoch
steps_per_epoch = len(train_generator)

# Train the model with callbacks
history = model.fit(
    train_generator,
    epochs=10,  # adjust as needed
    validation_data=validation_generator,
    callbacks=[model_checkpoint_callback]
)

# Save the trained weights after training
model.save('./Project Car Detection EfficientNet/EffNetB0.weight.h5')