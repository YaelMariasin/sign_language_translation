import numpy as np
import tensorflow as tf
from collections import Counter
import json
from create_database import read_all
from conver_json_to_vector import create_feature_vector
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, Conv3D, MaxPooling3D, GlobalAveragePooling3D, Input
)
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import pickle

def save_label_mapping(label_encoder, file_path):
    """Save the label encoder to a file."""
    with open(file_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"Label encoder saved to {file_path}")

# Function to load data from the database and create feature vectors
def load_data_from_db():
    """Load data from the database and transform JSON content into feature vectors."""
    vec_lst, label_lst = [], []
    not_processed_data = read_all()

    for id, label, category, json_data in not_processed_data:
        try:
            # Parse the JSON content into a dictionary
            json_dict = json.loads(json_data)

            # Create feature vector of shape (15000, 3)
            vectorized_data = create_feature_vector(json_dict)

            vec_lst.append(vectorized_data)
            label_lst.append(label)

        except Exception as e:
            print(f"Error processing data for ID {id}: {e}")
            continue

    # Extract only the main label (e.g., "label_1" -> "label")
    label_lst = [label.split("_")[0] for label in label_lst]

    # Add channel dimension to the features (for Conv3D compatibility)
    vec_lst = np.expand_dims(vec_lst, axis=-1)  # Shape becomes (batch_size, 15000, 3, 1)

    return np.array(vec_lst, dtype='float32'), label_lst

def compute_weights(y_train):
    """Compute class weights for imbalanced datasets."""
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    return dict(enumerate(class_weights))

def build_3d_cnn_model(input_shape, num_classes):
    """Build a 3D CNN model."""
    input_layer = Input(shape=input_shape)

    # 3D Convolutional Layers
    x = Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same')(input_layer)
    print(f"Shape after first Conv3D: {x.shape}")
    x = MaxPooling3D(pool_size=(2, 2, 1), padding='same')(x)  # Reduce height and width only
    print(f"Shape after first MaxPooling3D: {x.shape}")

    x = Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    print(f"Shape after second Conv3D: {x.shape}")
    x = MaxPooling3D(pool_size=(2, 2, 1), padding='same')(x)  # Reduce height and width only
    print(f"Shape after second MaxPooling3D: {x.shape}")

    x = Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    print(f"Shape after third Conv3D: {x.shape}")
    x = MaxPooling3D(pool_size=(2, 2, 1), padding='same')(x)  # Reduce height and width only
    print(f"Shape after third MaxPooling3D: {x.shape}")

    # Global pooling layer
    x = GlobalAveragePooling3D()(x)
    print(f"Shape after GlobalAveragePooling3D: {x.shape}")

    # Fully Connected Layer
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)

    # Output Layer
    output_layer = Dense(num_classes, activation='softmax')(x)

    return Model(inputs=input_layer, outputs=output_layer)


def train_model(model, X_train, y_train, class_weights):
    """Train the model."""
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)
    ]

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        class_weight=class_weights,
        callbacks=callbacks
    )

    # Plot training and validation accuracy
    plt.figure(figsize=(14, 6))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return history

def evaluate_model(model, X_test, y_test, label_encoder):
    """Evaluate the model."""
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    y_pred = np.argmax(model.predict(X_test), axis=-1)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

def create_model(pkl_file_name=None, model_filename=None):
    """Load data, build, train, and save the model."""
    features, labels = load_data_from_db()

    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    if not pkl_file_name:
        pkl_file_name = f'label_encoder_3d_cnn_{len(labels) // len(set(labels))}_vpw.pkl'

    save_label_mapping(label_encoder, pkl_file_name)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
    )

    # Verify distribution of labels
    train_label_distribution = Counter(y_train)
    test_label_distribution = Counter(y_test)
    print("Train Label Distribution:", train_label_distribution)
    print("Test Label Distribution:", test_label_distribution)

    # Compute class weights
    class_weights = compute_weights(y_train)

    # Build the model
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3], 1)  # Add channel dimension
    num_classes = len(label_encoder.classes_)
    model = build_3d_cnn_model(input_shape, num_classes)

    # Train the model
    history = train_model(model, X_train, y_train, class_weights)

    # Evaluate the model
    evaluate_model(model, X_test, y_test, label_encoder)

    # Save the model
    if not model_filename:
        videos_per_word = len(labels) // len(set(labels))
        model_filename = f'3d_cnn_on_{videos_per_word}_vpw.keras'

    if not model_filename.endswith('.keras'):
        model_filename = model_filename + '.keras'

    model.save(model_filename)
    print(f"Model saved as '{model_filename}'.")

if __name__ == "__main__":
    create_model()