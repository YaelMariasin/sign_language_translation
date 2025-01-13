import numpy as np
from sklearn.preprocessing import LabelEncoder


def classify_sign(numpy_object, model, label_encoder):
    """
    Classify a single ISL sign using the trained model.

    Args:
        numpy_object (numpy.ndarray): A numpy array of shape (15000, 3) representing the input features.
        model (tensorflow.keras.Model): The trained Keras model for classification.
        label_encoder (LabelEncoder): The label encoder used to encode the class labels.

    Returns:
        str: The predicted class label.
    """
    # Ensure the input shape is correct
    if numpy_object.shape != (15000, 3):
        raise ValueError(f"Invalid input shape: expected (15000, 3), got {numpy_object.shape}")

    # Add batch dimension to the input (expected shape: (1, 15000, 3))
    numpy_object = np.expand_dims(numpy_object, axis=0)

    # Predict the class probabilities
    predictions = model.predict(numpy_object)

    # Get the class with the highest probability
    predicted_class_index = np.argmax(predictions, axis=-1)[0]

    # Decode the class index to the label
    predicted_label = label_encoder.inverse_transform([predicted_class_index])[0]

    return predicted_label

test_sample = np.random.rand(15000, 3).astype('float32')  # Replace with your actual data
model_path = 'sign_language_video_model.h5'

labels = ["brother", "sister", 'cell', 'phone', 'welcome', 'word']
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

predicted_label = classify_sign(test_sample, model_path, label_encoder)
print(f"Predicted Label: {predicted_label}")