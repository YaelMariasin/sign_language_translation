import numpy as np
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import json
import pickle
from conver_json_to_vector import create_feature_vector
import json
from test_mediapipe import extract_motion_data, motion_data_to_json

def read_json_file(file_path):
    """
    Reads a JSON file and returns its content.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The parsed JSON content.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            json_content = json.load(file)
        return json_content
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON. {e}")
    return None


def load_label_mapping(file_path):
    """Load the label encoder from a file."""
    with open(file_path, 'rb') as f:
        label_encoder = pickle.load(f)
    print(f"Label encoder loaded from {file_path}")
    return list(label_encoder.classes_)

def classify_json_file(model_filename ,json_content, label_mapping):
    """
    Classifies a dummy matrix using a pre-trained model.

    Args:
        model_filename (str): Path to the saved model.
        input_shape (tuple): Shape of the input data (e.g., (50, 75, 3)).
        label_mapping (dict): Mapping of class indices to labels.

    Returns:
        str: Predicted class label.
    """
    # Load the saved model
    model = load_model(model_filename)

    input_matrix = create_feature_vector(json_content)
    # Assuming `input_data` is your input of shape (32, 75, 3)
    reshaped_data = input_matrix.reshape((1, 50, 75, 3))  # Add batch and time steps dimensions

    # Pass reshaped_data to the model
    predictions = model.predict(reshaped_data)
    predicted_class = np.argmax(predictions, axis=-1)[0]  # Get the predicted class index

    # Map the predicted index to the corresponding label
    predicted_label = label_mapping[predicted_class]

    return predicted_label

# Example usage
if __name__ == "__main__":
    test_original = True

    if test_original:
        folder_path_for_videos = 'sign_language_videos'
        folder_path_for_jsons = 'motion_data'
    else:
        folder_path_for_videos = 'test_videos/videos'
        folder_path_for_jsons = 'test_videos/jsons'
    for file_name in os.listdir(folder_path_for_videos):
        if file_name.split('.')[1] == "mp4":
            file_name = file_name.split('.')[0]

            trim_data = extract_motion_data(file_name, folder_name = folder_path_for_videos)
            motion_data_to_json(trim_data, file_name, folder_name = folder_path_for_jsons)

            json_content = read_json_file(f"{folder_path_for_jsons}/{file_name}.json")
            # Replace with your model file path
            model_filename = 'models/3d_rnn_cnn_on_23_vpw.keras'

            # Define a label mapping (example)
            label_encoder = load_label_mapping('models/label_encoder_3d_rnn_cnn_23_vpw.pkl')


            # Get the classification result
            predicted_label = classify_json_file(model_filename,json_content,label_encoder)
            print(f"Real label: {file_name}, Predicted Label: {predicted_label}")
