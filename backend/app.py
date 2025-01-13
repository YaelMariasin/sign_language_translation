from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import uuid
from werkzeug.utils import secure_filename

# Add the parent directory of the backend folder to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from test_mediapipe import extract_motion_data
from conver_json_to_vector import create_feature_vector

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the upload folder exists
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['video']
    if file and allowed_file(file.filename):
        filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        print(f"Video uploaded: {file_path}")

        data_frames = extract_motion_data("backend/"+ UPLOAD_FOLDER + "/" + filename)
        vector = create_feature_vector(data_frames)

        # Here, you would process the video with your ML model
        result = translate_sign_language(vector)

    
        return jsonify({"translation": result}), 200
    else:
        return jsonify({"error": "Invalid file format"}), 400

# Example function to simulate translation
def translate_sign_language(vector):
    # Logic to send the video to your ML model
    # Placeholder result for now
    return "Translated text for the uploaded sign language video."

if __name__ == '__main__':
    app.run(port=3000, debug=True)
