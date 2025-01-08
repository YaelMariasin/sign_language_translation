import cv2
import mediapipe as mp
import json
import numpy

# Global configuration
video_folder = "sign_language_videos/"
json_folder = "motion_data/"
output_folder = "generated_videos/"


def extract_motion_data(video_name):
    # Initialize MediaPipe pose and hands modules
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    # Load video
    video_path = "sign_language_videos/" + video_name + ".mp4"
    cap = cv2.VideoCapture(video_path)

    # Output file for motion data
    output_data = []

    # Initialize MediaPipe
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose, \
            mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to RGB (required by MediaPipe)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process pose
            pose_results = pose.process(frame_rgb)
            hands_results = hands.process(frame_rgb)

            # Extract key points
            frame_data = {"pose": [], "hands": []}
            if pose_results.pose_landmarks:
                frame_data["pose"] = [
                    {
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z,
                        "visibility": lm.visibility
                    } for lm in pose_results.pose_landmarks.landmark
                ]

            if hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    frame_data["hands"].append([
                        {"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand_landmarks.landmark
                    ])

            # Append to output data
            output_data.append(frame_data)

            # Optionally, draw landmarks on the frame for visualization
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            if hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Show the frame
            cv2.imshow('Sign Language Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    # Save motion data to a file
    json_path = "motion_data/" + video_name + ".json"
    with open(json_path, "w") as f:
        json.dump(output_data, f)

    print(f"Motion data saved to {json_path}")


def visualize_motion_data(video_name):
    """Visualize motion data from a JSON file."""
    # Load motion data
    json_path = json_folder + video_name + ".json"
    with open(json_path, "r") as f:
        motion_data = json.load(f)

    # Create a blank canvas for visualization
    canvas_size = (720, 1280, 3)  # Height, Width, Channels

    for frame_data in motion_data:
        canvas = numpy.ones(canvas_size, dtype=numpy.uint8) * 255  # White background

        # Draw pose landmarks
        if frame_data["pose"]:
            for lm in frame_data["pose"]:
                x, y = int(lm["x"] * canvas_size[1]), int(lm["y"] * canvas_size[0])
                cv2.circle(canvas, (x, y), 5, (0, 0, 255), -1)

        # Draw hand landmarks
        for hand_landmarks in frame_data["hands"]:
            for lm in hand_landmarks:
                x, y = int(lm["x"] * canvas_size[1]), int(lm["y"] * canvas_size[0])
                cv2.circle(canvas, (x, y), 5, (255, 0, 0), -1)

        # Show frame
        cv2.imshow('Visualizing Motion Data', canvas)
        if cv2.waitKey(100) & 0xFF == ord('q'):  # Adjust delay as needed
            break

    cv2.destroyAllWindows()


def visualize_as_stick_figure(video_name):
    """Visualize motion data as a stick figure."""
    # Load motion data
    json_path = json_folder + video_name + ".json"
    with open(json_path, "r") as f:
        motion_data = json.load(f)

    # Create a blank canvas for visualization
    canvas_size = (720, 1280, 3)  # Height, Width, Channels

    # Define connections for stick figure (based on MediaPipe connections)
    pose_connections = [
        (11, 13), (13, 15),  # Left arm
        (12, 14), (14, 16),  # Right arm
        (11, 12),            # Shoulders
        (23, 24),            # Hips
        (11, 23), (12, 24),  # Torso
        (23, 25), (25, 27),  # Left leg
        (24, 26), (26, 28)   # Right leg
    ]

    for frame_data in motion_data:
        canvas = numpy.ones(canvas_size, dtype=numpy.uint8) * 255  # White background

        # Draw pose landmarks and connections
        if frame_data["pose"]:
            landmarks = frame_data["pose"]
            for start, end in pose_connections:
                if start < len(landmarks) and end < len(landmarks):
                    x1, y1 = int(landmarks[start]["x"] * canvas_size[1]), int(landmarks[start]["y"] * canvas_size[0])
                    x2, y2 = int(landmarks[end]["x"] * canvas_size[1]), int(landmarks[end]["y"] * canvas_size[0])
                    cv2.line(canvas, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Draw connections

            # Draw individual points
            for lm in landmarks:
                x, y = int(lm["x"] * canvas_size[1]), int(lm["y"] * canvas_size[0])
                cv2.circle(canvas, (x, y), 5, (255, 0, 0), -1)

        # Show frame
        cv2.imshow('Stick Figure Animation', canvas)
        if cv2.waitKey(100) & 0xFF == ord('q'):  # Adjust delay as needed
            break

    cv2.destroyAllWindows()


def save_visualization_as_video(video_name):
    """Save the visualization as a video file."""
    # Load motion data
    json_path = json_folder + video_name + ".json"
    with open(json_path, "r") as f:
        motion_data = json.load(f)

    # Create a blank canvas for visualization
    canvas_size = (720, 1280, 3)  # Height, Width, Channels

    # Initialize video writer
    output_path = output_folder + video_name + "_recreated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 10  # Adjust frames per second
    out = cv2.VideoWriter(output_path, fourcc, fps, (canvas_size[1], canvas_size[0]))

    for frame_data in motion_data:
        canvas = numpy.ones(canvas_size, dtype=numpy.uint8) * 255  # White background

        # Draw pose landmarks
        if frame_data["pose"]:
            for lm in frame_data["pose"]:
                x, y = int(lm["x"] * canvas_size[1]), int(lm["y"] * canvas_size[0])
                cv2.circle(canvas, (x, y), 5, (0, 0, 255), -1)

        # Draw hand landmarks
        for hand_landmarks in frame_data["hands"]:
            for lm in hand_landmarks:
                x, y = int(lm["x"] * canvas_size[1]), int(lm["y"] * canvas_size[0])
                cv2.circle(canvas, (x, y), 5, (255, 0, 0), -1)

        # Write the frame to the video
        out.write(canvas)

    out.release()
    print(f"Recreated video saved to {output_path}")


if __name__ == "__main__":
    video_name = "word"  # Replace with the name of your video (without extension)

    # # Extract motion data from video
    # extract_motion_data(video_name)
    #
    # # Visualize motion data
    visualize_motion_data(video_name)
    #
    # # Save visualization as video
    # save_visualization_as_video(video_name)

    # visualize_as_stick_figure(video_name)
