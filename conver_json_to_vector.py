import os
import json
import numpy as np


def create_feature_vector(frames_data):
    # Limit frames_data to the first 200 frames if there are more
    max_frames = 200
    if len(frames_data) > max_frames:
        frames_data = frames_data[:max_frames]

    full_vector = []

    for frame in frames_data:
        full_vector.extend(extract_features(frame))

    # Ensure full_vector is a 2D array with shape (15000, 3)
    full_vector = np.array(full_vector)
    total_rows = 15000

    if len(full_vector) < total_rows:
        # Add rows of zeros to match the desired length
        padding = total_rows - len(full_vector)
        zero_padding = np.zeros((padding, 3))
        full_vector = np.vstack([full_vector, zero_padding])

    return full_vector

def extract_features(frame):
    vector = []

    # Process pose features
    pose_features = frame.get('pose', [])
    for feature in pose_features:
        x = feature.get('x', 0.0)
        y = feature.get('y', 0.0)
        z = feature.get('z', 0.0)
        vector.append(np.array([x, y, z]))

    # Process hand features
    hands = frame.get('hands', [])
    for hand_index in range(2):  # Ensure exactly two hands (or placeholders)
        if hand_index < len(hands):
            hand_features = hands[hand_index]
            for feature in hand_features:
                x = feature.get('x', 0.0)
                y = feature.get('y', 0.0)
                z = feature.get('z', 0.0)
                vector.append(np.array([x, y, z]))
        else:
            # Add placeholder for missing hand
            vector.extend([np.array([0.0, 0.0, 0.0])] * 21)

    return vector