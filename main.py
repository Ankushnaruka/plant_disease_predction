import os
import kagglehub
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
import numpy as np

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# ---- 1. Download dataset ----
print("Downloading dataset...")
path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
print(f"Dataset downloaded to: {path}")

# ---- 2. Configuration ----
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
EPOCHS = 15

# Check directory structure
train_dir = os.path.join(path, "New Plant Diseases Dataset(Augmented)", "New Plant Diseases Dataset(Augmented)", "train")
val_dir = os.path.join(path, "New Plant Diseases Dataset(Augmented)", "New Plant Diseases Dataset(Augmented)", "valid")
test_dir = os.path.join(path, "test")

# Alternative path structure (in case the above doesn't work)
if not os.path.exists(train_dir):
    train_dir = os.path.join(path, "train")
    val_dir = os.path.join(path, "valid")
    test_dir = os.path.join(path, "test")

print(f"Train directory: {train_dir}")
print(f"Validation directory: {val_dir}")
print(f"Test directory: {test_dir}")

# Verify directories exist
for dir_path, name in [(train_dir, "train"), (val_dir, "validation"), (test_dir, "test")]:
    if os.path.exists(dir_path):
        print(f"✓ {name} directory found")
        print(f"  Classes: {len(os.listdir(dir_path))}")
    else:
        print(f"✗ {name} directory not found: {dir_path}")

# ---- 3. Load datasets with improved preprocessing ----
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
    validation_split=None,
    label_mode='categorical'  # Use categorical for better compatibility
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
    validation_split=None,
    label_mode='categorical'
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
    validation_split=None,
    label_mode='categorical'
)

# Get class names and number of classes
class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Number of classes: {num_classes}")
print(f"Class names: {class_names[:]}...")  # Show first 5 classes

# ---- 4. Data preprocessing and augmentation ----
AUTOTUNE = tf.data.AUTOTUNE

# Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

# Apply preprocessing
def prepare_dataset(ds, shuffle=False, augment=False):
    # Normalize pixel values to [0,1]
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
    
    if shuffle:
        ds = ds.shuffle(1000)
    
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y))
    
    return ds.prefetch(buffer_size=AUTOTUNE)

train_ds = prepare_dataset(train_ds, shuffle=True, augment=True)
val_ds = prepare_dataset(val_ds)
test_ds = prepare_dataset(test_ds)

# ---- 5. Build improved model with transfer learning ----
def create_model(num_classes, use_transfer_learning=True):
    if use_transfer_learning:
        # Use MobileNetV2 as base model
        base_model = MobileNetV2(
            input_shape=IMG_SIZE + (3,),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False  # Freeze base model
        
        model = tf.keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])
    else:
        # Custom CNN model
        model = models.Sequential([
            layers.Input(shape=IMG_SIZE+(3,)),
            
            # First Conv Block
            layers.Conv2D(32, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            
            # Second Conv Block
            layers.Conv2D(64, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            
            # Third Conv Block
            layers.Conv2D(128, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            
            # Fourth Conv Block
            layers.Conv2D(256, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            
            # Classifier
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
    
    return model

# Create model
model = create_model(num_classes, use_transfer_learning=True)

# ---- 6. Compile model (FIXED: removed top_5_accuracy) ----
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel Summary:")
model.summary()

# ---- 7. Setup callbacks ----
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-7
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_plant_disease_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False
    )
]

# ---- 8. Train the model ----
print(f"\nStarting training for {EPOCHS} epochs...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# ---- 9. Evaluate the model (FIXED: removed top_5_accuracy) ----
print("\nEvaluating on test set...")
test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
print(f"Test Accuracy: {test_accuracy:.4f}")

# ---- 10. Plot training history ----
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_training_history(history)

# ---- 11. Save the final model ----
model.save("plant_disease_classifier_final.h5")
print("\nModel saved as 'plant_disease_classifier_final.h5'")

# ---- 12. Create a prediction function ----
def predict_disease(model, image_path, class_names):
    """
    Predict plant disease from an image
    """
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch dimension
    img_array = img_array / 255.0  # Normalize
    
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))
    
    return predicted_class, confidence

# Save class names for later use
import pickle
with open('class_names.pkl', 'wb') as f:
    pickle.dump(class_names, f)

print("\nClass names saved as 'class_names.pkl'")
print(f"Total classes trained: {len(class_names)}")
print("\nTraining completed successfully!")

# Show final results
print(f"\nFinal Results:")
print(f"- Best model saved: best_plant_disease_model.h5")
print(f"- Final model saved: plant_disease_classifier_final.h5")
print(f"- Test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"- Training history plot saved: training_history.png")