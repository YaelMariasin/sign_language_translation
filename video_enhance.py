import cv2
import mediapipe as mp
import numpy as np

def process_video(input_path, output_path):
    # Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {input_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Initialize the video writer
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Mediapipe Pose and Hand solutions
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    pose = mp_pose.Pose()
    hands = mp_hands.Hands()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame for a mirror-like view
        frame = cv2.flip(frame, 1)

        # Enhance the frame
        enhanced_frame = enhance_frame(frame, pose, hands)

        # Write the enhanced frame to the output video
        out.write(enhanced_frame)

    # Release resources
    cap.release()
    out.release()
    print(f"Enhanced video saved to {output_path}")


def enhance_frame(frame, pose, hands):
    # Convert the frame to RGB (required by Mediapipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect pose and hands
    pose_results = pose.process(rgb_frame)
    hands_results = hands.process(rgb_frame)

    # Create a black background
    black_background = np.zeros_like(frame)

    # Highlight the body if detected
    if pose_results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            black_background,
            pose_results.pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=4, circle_radius=3),
            mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)
        )

    # Highlight the hands if detected
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                black_background,
                hand_landmarks,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=4, circle_radius=3),
                mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2)
            )

    # Combine the enhanced person with the original background
    combined_frame = cv2.addWeighted(frame, 0.5, black_background, 1.0, 0)

    return combined_frame


# Input and output paths
input_video_path = "sign_language_videos/welcome.mp4"
output_video_path = "sign_language_videos/welcome_enhanced_highlighted_video.mp4"

# Process the video
process_video(input_video_path, output_video_path)