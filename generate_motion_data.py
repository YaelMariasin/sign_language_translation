import os
import json
import random
import cv2
import numpy
import copy

# Create directories if they don't exist
os.makedirs("generated_motion_data", exist_ok=True)

folder_with_new_jsons = "generated_motion_data"
folder_with_original_jsons = "motion_data"


def load_json(file_name):
    """
    Load a JSON file from the motion_data folder.

    Args:
        file_name (str): The name of the JSON file to load.

    Returns:
        dict: The loaded JSON data.
    """
    with open(os.path.join(folder_with_original_jsons, file_name), "r") as file:
        return json.load(file)


def save_json(data, file_name):
    """
    Save a JSON file to the generated_motion_data folder.

    Args:
        data (dict): The JSON data to save.
        file_name (str): The name of the output JSON file.
    """
    with open(os.path.join(folder_with_new_jsons, file_name), "w") as file:
        json.dump(data, file, indent=4)


def shift_position(data, shift_x=0, shift_y=0):
    """
    Shift all landmarks in the JSON data by a given x and y offset.

    Args:
        data (list): The JSON data representing motion frames.
        shift_x (float): The amount to shift landmarks along the x-axis.
        shift_y (float): The amount to shift landmarks along the y-axis.

    Returns:
        list: The modified JSON data with shifted positions.
    """
    for frame in data:
        for landmark in frame['pose']:
            landmark['x'] += shift_x
            landmark['y'] += shift_y
        for hand_landmarks in frame.get('hands', []):
            for landmark in hand_landmarks:
                landmark['x'] += shift_x
                landmark['y'] += shift_y
    return data


def scale_motion(data, scale_factor):
    """
    Scale all landmarks in the JSON data by a given factor.

    Args:
        data (list): The JSON data representing motion frames.
        scale_factor (float): The factor by which to scale the landmarks.

    Returns:
        list: The modified JSON data with scaled positions.
    """
    for frame in data:
        for landmark in frame['pose']:
            landmark['x'] *= scale_factor
            landmark['y'] *= scale_factor
            landmark['z'] *= scale_factor
        for hand_landmarks in frame.get('hands', []):
            for landmark in hand_landmarks:
                landmark['x'] *= scale_factor
                landmark['y'] *= scale_factor
                landmark['z'] *= scale_factor
    return data


def add_noise(data, noise_level):
    """
    Add random noise to all landmarks in the JSON data.

    Args:
        data (list): The JSON data representing motion frames.
        noise_level (float): The maximum amount of random noise to add to each coordinate.

    Returns:
        list: The modified JSON data with added noise.
    """
    const_make_the_noise_smaller = 100
    for frame in data:
        for landmark in frame['pose']:
            landmark['x'] += random.uniform(-noise_level, noise_level / const_make_the_noise_smaller)
            landmark['y'] += random.uniform(-noise_level, noise_level / const_make_the_noise_smaller)
            landmark['z'] += random.uniform(-noise_level, noise_level / const_make_the_noise_smaller)
        for hand_landmarks in frame.get('hands', []):
            for landmark in hand_landmarks:
                landmark['x'] += random.uniform(-noise_level, noise_level / const_make_the_noise_smaller)
                landmark['y'] += random.uniform(-noise_level, noise_level / const_make_the_noise_smaller)
                landmark['z'] += random.uniform(-noise_level, noise_level / const_make_the_noise_smaller)
    return data


def add_up_down_movement(data, shift_y):
    """
    Add vertical movement (up or down) to all landmarks.

    Args:
        data (list): The JSON data representing motion frames.
        shift_y (float): The amount to shift landmarks along the y-axis.

    Returns:
        list: The modified JSON data with vertical movement.
    """
    return shift_position(data, shift_x=0, shift_y=shift_y)


def generate_variations(file_name):
    """
    Generate new variations of the motion JSON data with multiple changes.

    Args:
        file_name (str): The base name of the input JSON file (without .json).
    """
    data = load_json(f"{file_name}.json")
    # Single transformations
    shifted_right = shift_position(copy.deepcopy(data), shift_x=0.1,
                                   shift_y=0.0)  # Shift landmarks slightly to the right
    save_json(shifted_right, f"{file_name}_shifted_right.json")

    shifted_far_right = shift_position(copy.deepcopy(data), shift_x=0.3,
                                       shift_y=0.0)  # Shift landmarks farther to the right
    save_json(shifted_far_right, f"{file_name}_shifted_far_right.json")

    shifted_left = shift_position(copy.deepcopy(data), shift_x=-0.1,
                                  shift_y=0.0)  # Shift landmarks slightly to the left
    save_json(shifted_left, f"{file_name}_shifted_left.json")

    scaled_up = scale_motion(copy.deepcopy(data), scale_factor=1.2)  # Scale up (zoom in)
    save_json(scaled_up, f"{file_name}_scaled_up.json")

    scaled_down = scale_motion(copy.deepcopy(data), scale_factor=0.8)  # Scale down (zoom out)
    save_json(scaled_down, f"{file_name}_scaled_down.json")

    noisy_data = add_noise(copy.deepcopy(data), noise_level=0.01)  # Add very small random noise
    save_json(noisy_data, f"{file_name}_noisy.json")

    moved_up = add_up_down_movement(copy.deepcopy(data), shift_y=0.1)  # Shift landmarks slightly upward
    save_json(moved_up, f"{file_name}_moved_up.json")

    moved_down = add_up_down_movement(copy.deepcopy(data), shift_y=-0.1)  # Shift landmarks slightly downward
    save_json(moved_down, f"{file_name}_moved_down.json")

    # Combined transformations
    combined_1 = add_noise(scale_motion(shift_position(copy.deepcopy(data), shift_x=0.05, shift_y=0.05), 1.1), 0.005)
    save_json(combined_1, f"{file_name}_combined_1.json")  # Slightly shifted, scaled up, and very small noise

    combined_2 = add_noise(scale_motion(shift_position(copy.deepcopy(data), shift_x=-0.05, shift_y=-0.05), 0.9), 0.005)
    save_json(combined_2, f"{file_name}_combined_2.json")  # Slightly shifted, scaled down, and very small noise


def visualize_motion_data(file_name):
    """
    Visualize motion data from a JSON file.

    Args:
        file_name (str): The name of the JSON file to visualize.
    """
    json_path = os.path.join(folder_with_new_jsons, file_name)
    with open(json_path, "r") as f:
        motion_data = json.load(f)

    canvas_size = (720, 1280, 3)  # Canvas dimensions (Height, Width, Channels)

    for frame_data in motion_data:
        canvas = numpy.ones(canvas_size, dtype=numpy.uint8) * 255  # White background

        # Draw pose landmarks
        for lm in frame_data.get("pose", []):
            x, y = int(lm["x"] * canvas_size[1]), int(lm["y"] * canvas_size[0])
            cv2.circle(canvas, (x, y), 5, (0, 0, 255), -1)

        # Draw hand landmarks
        for hand_landmarks in frame_data.get("hands", []):
            for lm in hand_landmarks:
                x, y = int(lm["x"] * canvas_size[1]), int(lm["y"] * canvas_size[0])
                cv2.circle(canvas, (x, y), 5, (255, 0, 0), -1)

        # Add title with file name
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, file_name, (50, 50), font, 1, (0, 0, 0), 2, cv2.LINE_AA)

        # Show frame
        cv2.imshow('Visualizing Motion Data', canvas)
        if cv2.waitKey(100) & 0xFF == ord('q'):  # Adjust delay as needed
            break

    cv2.destroyAllWindows()


def visualize_all_files(input_file_name):
    """
    Visualize all generated motion JSON files starting with the input file name.

    Args:
        input_file_name (str): The base name of the input file.
    """
    for file_name in os.listdir(folder_with_new_jsons):
        if file_name.startswith(input_file_name) and file_name.endswith(".json"):
            visualize_motion_data(file_name)


if __name__ == "__main__":
    input_file_name = "word"  # Replace with the desired file name (without .json extension)
    generate_variations(input_file_name)
    visualize_all_files(input_file_name)
