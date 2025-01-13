import os
import json
import numpy as np


# Function to extract features from a frame
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


# Process all frames
def create_feature_vector(frames_data):
    full_vector = []

    for frame in frames_data:
        full_vector.extend(extract_features(frame))

    return np.array(full_vector)





