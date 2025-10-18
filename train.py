# ==============================================================================
# 1. IMPORTS: Load all the necessary tools
# ==============================================================================
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import os

# ==============================================================================
# --- IMPORTANT: CHANGE THESE THREE VALUES ---
# ==============================================================================
# 2. CONFIGURATION: Set up the paths to your data and key parameters
# ==============================================================================

# --- Path to the folder containing your training images ---
train_dir = 'train_data' 

# --- Path to the folder containing your validation images ---
validation_dir = 'validation_data'

# --- The total number of different bird species (folders) you have ---
NUMBER_OF_BIRD_CLASSES = 200 

# You can also change these if you want
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20 # 10 is a good start, you can increase this later for better accuracy
# ==============================================================================


# ==============================================================================
# 3. DATA PREPARATION: Prepare the images for the model
# This is the "recipe" for preparing the image "ingredients"
# ==============================================================================

# This is the special preparation function required for the VGG16 model.
from tensorflow.keras.applications.vgg16 import preprocess_input

# Create an "Image Data Generator" for training images.
# This will create new, slightly different versions of your images during training
# to make the model better (this is called data augmentation).
print("Setting up Training Data Generator...")
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input, # <-- The ONLY preparation step needed.
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Create an "Image Data Generator" for validation images.
# For validation, we ONLY need to prepare the image. No changes or augmentation.
print("Setting up Validation Data Generator...")
validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Connect the generators to your actual image folders
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print("Data generators are ready.")

# ==============================================================================
# 4. MODEL BUILDING: Create the "chef" (the neural network)
# ==============================================================================
print("Building the model...")

# Load the VGG16 model from Keras, without its final classification layers ("include_top=False").
# This is our powerful base model.
base_model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# Freeze the layers of the base VGG16 model so their learned weights don't change.
for layer in base_model.layers:
    layer.trainable = False

# Now, let's create our own new classification layers (the "head") to put on top.
head_model = base_model.output
head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
head_model = Flatten(name="flatten")(head_model)
head_model = Dense(128, activation="relu")(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(NUMBER_OF_BIRD_CLASSES, activation="softmax")(head_model)

# Put the base model and our new head together to create the final model.
model = Model(inputs=base_model.input, outputs=head_model)

print("Model has been built.")

# ==============================================================================
# 5. MODEL COMPILATION: Prepare the model for training
# ==============================================================================
print("Compiling the model...")

# Set a learning rate for the optimizer
learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate)

# Compile the model with loss function, optimizer, and metrics
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

print("Model has been compiled.")
model.summary()

# ==============================================================================
# 6. TRAINING: Start the actual training process
# ==============================================================================
print(f"Starting training for {EPOCHS} epochs...")

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=EPOCHS
)

# ==============================================================================
# 7. SAVING: Save the final, trained model to a file
# ==============================================================================
# Save the model to a file so you can use it in your app
model.save("birds_new_and_improved.h5")

print("\n" + "="*50)
print(f"TRAINING COMPLETE! 🚀")
print(f"Your new model has been saved as 'birds_new_and_improved.h5'")
print("You can now use this file in your Flask web application.")
print("="*50)