from video_augmentation import augment_all_files
import shutil
import os
from test_mediapipe import create_original_motion_data
from create_database import create_db, drop_video_data_table, add_json_files_from_folder
# from models.model_3d_cnn_rnn import create_model
from models.model_3d_cnn import create_model
from classification import classify
from pose_format_augmentation import generate_augmented_json_files

# Path settings
original_folder = "motion_data/"
augmented_folder = "generated_motion_data/"
csv_file_path = "augmented_data_summary.csv"  # Output CSV file

os.makedirs(augmented_folder, exist_ok=True)
AMOUNT_OF_VARIATIONS = 15

def create_original_jsons():
    os.makedirs(augmented_folder, exist_ok=True)
    create_original_motion_data()

def create_augmentations(folder_path ="generated_motion_data"):
    print(f"Start creating augmented data at {folder_path}")
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)
    augment_all_files(AMOUNT_OF_VARIATIONS)
    generate_augmented_json_files("motion_data", AMOUNT_OF_VARIATIONS)

def delete_folder_content(folder_path):
    # Path to the folder

    # Delete the folder
    try:
        shutil.rmtree(folder_path)
        print(f"Old augmentations were deleted at {folder_path}")
    except OSError as e:
        print(f"Error: {e.strerror}")

def create_database_and_table():
    create_db()

def drop_db_table():
    drop_video_data_table()

def main():
    pkl_file_name = f"label_encoder_3d_cnn_{AMOUNT_OF_VARIATIONS}_vpw"
    model_filename = f"3d_cnn_on_{AMOUNT_OF_VARIATIONS}_vpw"

    # delete_folder_content("motion_data")
    # create_original_jsons()
    # delete_folder_content("generated_motion_data")
    # create_augmentations()
    # drop_db_table()
    # create_database_and_table()
    # add_json_files_from_folder("generated_motion_data", "first_model")
    # add_json_files_from_folder("pose_format_augmentation", "first_model")
    create_model(pkl_file_name=pkl_file_name, model_filename=model_filename)
    results_list = classify(pkl_file_name, model_filename, test_original = True)
    # for result in results_list:
    #     print(result)
    # Calculate the maximum width for the "Real label" column
    max_real_label_length = max(len(real_label) for real_label, _ in results_list)

    # Print the aligned results
    for real_label, predicted_label in results_list:
        print(f"Real label: {real_label.ljust(max_real_label_length)} | Predicted Label: {predicted_label}")
    results_list = classify(pkl_file_name, model_filename, test_original=False)
    max_real_label_length = max(len(real_label) for real_label, _ in results_list)

    for real_label, predicted_label in results_list:
        print(f"Real label: {real_label.ljust(max_real_label_length)} | Predicted Label: {predicted_label}")

main()