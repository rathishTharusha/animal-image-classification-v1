# %% [markdown]
# ### **Animal Image Classifier using CNN**

# %% [markdown]
# **1. Data Preparation & Exploration**
# - Download and extract the dataset (from Kaggle or your local folder).
# - Organize images into train/validation/test folders by class.
# - Visualize sample images from each class.
# - Perform data augmentation to increase robustness.. Data Preparation & Exploration

# %%
import os
import shutil
import random
from glob import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

import matplotlib.pyplot as plt

# Set paths
base_dir = 'dataset/Animals'
classes = ['cats', 'dogs', 'snakes']
output_base = 'dataset/organized'

# Create train/val/test split
split_ratios = {'train': 0.8, 'val': 0.1, 'test': 0.1}

# Organize images into train/val/test folders
if not os.path.exists(output_base):
    for cls in classes:
        img_paths = glob(os.path.join(base_dir, cls, '*'))
        random.shuffle(img_paths)
        n_total = len(img_paths)
        n_train = int(n_total * split_ratios['train'])
        n_val = int(n_total * split_ratios['val'])
        splits = {
            'train': img_paths[:n_train],
            'val': img_paths[n_train:n_train + n_val],
            'test': img_paths[n_train + n_val:]
        }
        for split, files in splits.items():
            split_dir = os.path.join(output_base, split, cls)
            os.makedirs(split_dir, exist_ok=True)
            for f in files:
                shutil.copy(f, split_dir)

# Visualize sample images from each class
fig, axes = plt.subplots(1, len(classes), figsize=(12, 4))
for idx, cls in enumerate(classes):
    sample_dir = os.path.join(output_base, 'train', cls)
    sample_img = random.choice(os.listdir(sample_dir))
    img_path = os.path.join(sample_dir, sample_img)
    img = load_img(img_path, target_size=(128, 128))
    axes[idx].imshow(img)
    axes[idx].set_title(cls)
    axes[idx].axis('off')
plt.tight_layout()
plt.show()

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Visualize augmented images for one sample
sample_dir = os.path.join(output_base, 'train', classes[0])
sample_img = random.choice(os.listdir(sample_dir))
img_path = os.path.join(sample_dir, sample_img)
img = load_img(img_path, target_size=(128, 128))
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

fig, ax = plt.subplots(1, 5, figsize=(15, 3))
i = 0
for batch in datagen.flow(x, batch_size=1):
    ax[i].imshow(batch[0].astype('uint8'))
    ax[i].axis('off')
    i += 1
    if i == 5:
        break
plt.suptitle(f'Augmented samples: {classes[0]}')
plt.show()

# %% [markdown]
# **2. Data Loading with Keras**
# - Use `ImageDataGenerator` for loading and augmenting images.
# - Split data into training, validation, and test sets.

# %%
# Prepare ImageDataGenerators for train, validation, and test sets
train_dir = os.path.join(output_base, 'train')
val_dir = os.path.join(output_base, 'val')
test_dir = os.path.join(output_base, 'test')

img_height, img_width = 128, 128  # Standard for transfer learning
batch_size = 32  # batch_size means number of images processed together 

# Training data generator with augmentation
train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Validation and test generators without augmentation
test_val_datagen = ImageDataGenerator()
val_gen = test_val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)
test_gen = test_val_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# %% [markdown]
# **3. Load Pre-trained ResNet Model**
# - Use ResNet50 (or ResNet101 for more complexity) with imagenet weights.
# - Exclude the top layer (include_top=False) to add custom layers.

# %%
from tensorflow.keras.applications import ResNet50

# Load ResNet50 with imagenet weights, excluding the top layer
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False  # Freeze base model layers

base_model.summary()

# %% [markdown]
# **4. Add Custom Layers**
# - Add Global Average Pooling, Dropout, Dense layers, and a final softmax output layer.

# %%
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model

# Add custom layers on top of the base_model
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
output = Dense(len(classes), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.summary()

# %% [markdown]
# **5. Freeze Pre-trained Layers**
# - Freeze all layers in the ResNet base to train only the custom head.

# %%
for layer in base_model.layers:
    layer.trainable = False

# Optionally, check if all layers are frozen
print("All base_model layers frozen:", all(not layer.trainable for layer in base_model.layers))

# %% [markdown]
# **6. Compile and Train the Model (Initial Training)**
# - Use Adam optimizer, categorical crossentropy, and accuracy metrics.
# - Add callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau.

# %%
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Define callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
]

# Train the model
history = model.fit(
    train_gen,
    epochs=25,
    validation_data=val_gen,
    callbacks=callbacks
)

# %% [markdown]
# **7. Fine-tune the Model**
# - Unfreeze some top layers of ResNet for fine-tuning.
# - Retrain with a lower learning rate.

# %%
# Unfreeze the top layers of the base_model for fine-tuning
for layer in base_model.layers[-10:]:
    layer.trainable = True

# Recompile the model with a lower learning rate
model.compile(
    optimizer=Adam(learning_rate=1e-6),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Retrain the model
fine_tune_history = model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen,
    callbacks=callbacks
)

# %% [markdown]
# **8. Evaluate and Visualize**
# - Plot training/validation accuracy and loss.
# - Show confusion matrix and classification report.
# - Visualize predictions on test images.

# %%
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns

# Plot training/validation accuracy and loss
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].plot(history.history['accuracy'], label='Train Acc')
axs[0].plot(history.history['val_accuracy'], label='Val Acc')
if 'accuracy' in fine_tune_history.history:
    axs[0].plot(np.arange(len(history.history['accuracy']), len(history.history['accuracy']) + len(fine_tune_history.history['accuracy'])), fine_tune_history.history['accuracy'], label='Train Acc (ft)')
    axs[0].plot(np.arange(len(history.history['val_accuracy']), len(history.history['val_accuracy']) + len(fine_tune_history.history['val_accuracy'])), fine_tune_history.history['val_accuracy'], label='Val Acc (ft)')
axs[0].set_title('Accuracy')
axs[0].legend()
axs[1].plot(history.history['loss'], label='Train Loss')
axs[1].plot(history.history['val_loss'], label='Val Loss')
if 'loss' in fine_tune_history.history:
    axs[1].plot(np.arange(len(history.history['loss']), len(history.history['loss']) + len(fine_tune_history.history['loss'])), fine_tune_history.history['loss'], label='Train Loss (ft)')
    axs[1].plot(np.arange(len(history.history['val_loss']), len(history.history['val_loss']) + len(fine_tune_history.history['val_loss'])), fine_tune_history.history['val_loss'], label='Val Loss (ft)')
axs[1].set_title('Loss')
axs[1].legend()
plt.show()

# Predict on test set
y_true = test_gen.classes
y_pred_probs = model.predict(test_gen)
y_pred = np.argmax(y_pred_probs, axis=1)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Classification report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=classes))

# Visualize predictions on test images
test_gen.reset()
fig, axes = plt.subplots(2, 5, figsize=(18, 8))
for i, ax in enumerate(axes.flat):
    img, label = next(test_gen)
    pred_idx = np.argmax(model.predict(img), axis=1)[0]
    true_idx = np.argmax(label, axis=1)[0]
    ax.imshow(img[0].astype('uint8'))
    ax.set_title(f"True: {classes[true_idx]}\nPred: {classes[pred_idx]}")
    ax.axis('off')
plt.tight_layout()
plt.show()

# %% [markdown]
# **9. Save and Export the Model**

# %%
# Save the trained model in HDF5 format
model.save('final_model.h5')

print("Model saved as 'final_model.h5'")


