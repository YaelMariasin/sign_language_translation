import os
from moviepy import *
from moviepy.editor import *


def crop_video(video_name, x, y, width, height):
    try:
        # Define folder paths
        base_folder = os.path.dirname(__file__)  # Get the current script's directory
        input_folder = os.path.join(base_folder, "original_videos")
        output_folder = os.path.join(base_folder, "cropped_videos")

        # Build input and output file paths
        input_path = os.path.join(input_folder, video_name)
        output_path = os.path.join(output_folder, f"cropped_{video_name}")

        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Load video
        video_clip = VideoFileClip(input_path)

        # Crop the video
        # cropped_clip = video_clip.crop(x1=x, y1=y, width=width, height=height)
        cropped_clip = video_clip.crop(x1=x, y1=y, x2=x + width, y2=y + height)

        # Write the cropped video to the output path
        cropped_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

        print(f"Video successfully cropped and saved at: {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # Define folder paths
    base_folder = os.path.dirname(__file__)
    input_folder = os.path.join(base_folder, "original_videos")

    # Get video name from user
    video_name = "video1.mp4"
    input_path = os.path.join(input_folder, video_name)

    try:
        # Load the video to get its size
        video_clip = VideoFileClip(input_path)
        width = video_clip.size[0]
        height = video_clip.size[1]
        print(f"Video duration: {video_clip.duration} seconds")
        print(f"Original video size: {width}x{height} pixels")
    except Exception as e:
        print(f"Could not load video: {e}")
        exit()

    # Get cropping parameters from user (or set default values)
    x = 50
    y = 50
    width_crop = 200
    height_crop = 200

    # Crop the video
    crop_video(video_name, x, y, width_crop, height_crop)
