import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, LSTM, Dropout, Bidirectional, BatchNormalization,
    Conv1D, Input, GlobalAveragePooling1D
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# Check GPU availability
print("Available GPUs:", tf.config.list_physical_devices('GPU'))

# Load and preprocess the data
def load_data(csv_file):
    """
    Load and preprocess data from a CSV file with individual feature columns.
    Each video will have its frames aggregated into a consistent 3D feature array.
    Args:
        csv_file (str): Path to the CSV file.
    Returns:
        tuple: (features, labels)
    """
    data = pd.read_csv(csv_file)
    print("CSV Columns:", data.columns)

    # Extract feature column prefixes
    pose_columns = [col for col in data.columns if col.startswith('pose_')]
    left_hand_columns = [col for col in data.columns if col.startswith('left_hand_')]
    right_hand_columns = [col for col in data.columns if col.startswith('right_hand_')]

    features = []
    labels = []

    # Group data by 'File Name' (each video)
    video_groups = data.groupby('File Name')

    for video_name, group in video_groups:
        video_features = []

        for _, row in group.iterrows():
            try:
                # Extract features for each frame and ensure numeric conversion
                pose_features = np.array(row[pose_columns], dtype='float32')
                left_hand_features = np.array(row[left_hand_columns], dtype='float32')
                right_hand_features = np.array(row[right_hand_columns], dtype='float32')

                # Concatenate all features for this frame
                frame_features = np.concatenate([pose_features, left_hand_features, right_hand_features])
                video_features.append(frame_features)

            except ValueError as e:
                print(f"Skipping malformed row in video {video_name}: {e}")
                continue

        # Append to the list of features and labels if video_features is valid
        if video_features:
            features.append(np.array(video_features, dtype='float32'))
            labels.append(video_name)

    return features, labels

# Load and aggregate data
features, labels = load_data("augmented_data_summary.csv")

# Pad sequences to ensure same number of frames per video
max_frames = 100  # Adjust based on your dataset
padded_features = pad_sequences(features, padding='post', dtype='float32', maxlen=max_frames)

# Reshape features for scaling
num_features = padded_features.shape[2]
padded_features = padded_features.reshape(-1, num_features)

# Standardize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(padded_features)

# Reshape back to 3D (video count, max_frames, num_features)
scaled_features = scaled_features.reshape(-1, max_frames, num_features)

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    scaled_features, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
)

# Compute class weights for imbalanced data
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

# Learning rate scheduler
def lr_scheduler(epoch, lr):
    return lr * tf.math.exp(-0.05) if epoch >= 10 else lr

# Build the model
def create_model(input_shape, num_classes):
    input_layer = Input(shape=input_shape)

    # Convolutional layer
    x = Conv1D(64, kernel_size=3, activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Bidirectional LSTM
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Global Average Pooling
    x = GlobalAveragePooling1D()(x)

    # Fully connected layer
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Output layer
    output_layer = Dense(num_classes, activation='softmax')(x)

    return Model(inputs=input_layer, outputs=output_layer)

# Create and compile the model
model = create_model(input_shape=(max_frames, num_features), num_classes=len(label_encoder.classes_))
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Custom callback to print progress
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1}: Loss={logs['loss']:.4f}, Accuracy={logs['accuracy']:.4f}, "
              f"Val Loss={logs['val_loss']:.4f}, Val Accuracy={logs['val_accuracy']:.4f}")

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5),
    tf.keras.callbacks.LearningRateScheduler(lr_scheduler),
    CustomCallback()
]

# Train the model
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    class_weight=class_weights
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Generate predictions and classification report
y_pred = np.argmax(model.predict(X_test), axis=-1)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save the model
model.save('sign_language_video_model.h5')
print("Model saved as 'sign_language_video_model.h5'.")

# Plot learning curves
plt.figure(figsize=(14, 6))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()