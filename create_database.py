import os
import json
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv
from datetime import datetime
from sqlalchemy.sql import text

# Load environment variables from the .env file
load_dotenv()

# Get database credentials from environment variables
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME')

# Database connection configuration
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)
Base = declarative_base()

# Define the VideoData table
class VideoData(Base):
    __tablename__ = 'video_data'

    id = Column(Integer, primary_key=True, autoincrement=True)  # Auto-increment primary key
    label = Column(String(255), nullable=False)  # Label for the word
    category = Column(String(100), nullable=True)  # Category of the word
    json_output = Column(Text, nullable=True)  # JSON output

# Create a database session
Session = sessionmaker(bind=engine)
session = Session()

# Define the VideoUpload table
class VideoUpload(Base):
    __tablename__ = 'video_uploads'

    id = Column(Integer, primary_key=True, autoincrement=True)  # Auto-increment primary key
    timestamp = Column(DateTime, default=datetime.utcnow)  # Timestamp for when the video was uploaded
    label = Column(String(255), nullable=True)  # Label for the video (can be empty initially)
    json_output = Column(Text, nullable=True)  # JSON output from MediaPipe

# Function to extract label up to the first underscore
def extract_label(file_name):
    return file_name.split('_')[0]

# Function to add a single JSON file to the database
def add_json_file(file_path, category):
    try:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        label = file_name
        with open(file_path, 'r', encoding='utf-8') as f:
            json_output = json.load(f)
        new_video = VideoData(
            label=label,
            category=category,
            json_output=json.dumps(json_output)  # Store JSON as a string
        )
        session.add(new_video)
        session.commit()
        print(f"Added JSON file '{file_path}' to the database with ID: {new_video.id}")
    except Exception as e:
        session.rollback()
        print(f"Failed to add JSON file '{file_path}': {e}")

# Function to add a video upload entry
def add_video_upload(json_data, label=None):
    try:
        new_upload = VideoUpload(
            label=label,
            json_output=json.dumps(json_data)  # Store JSON as a string
        )
        session.add(new_upload)
        session.commit()
        print(f"Added video upload to the database with ID: {new_upload.id}")
    except Exception as e:
        session.rollback()
        print(f"Failed to add video upload: {e}")

# Function to add all JSON files from a folder to the database
def add_json_files_from_folder(folder_path, category):
    try:
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.json'):
                file_path = os.path.join(folder_path, file_name)
                add_json_file(file_path, category)
    except Exception as e:
        print(f"Failed to add JSON files from folder '{folder_path}': {e}")


# Function to read all records from the database and return a list of lists
def read_all():
    try:
        records = session.query(VideoData).all()
        result = [[record.id, record.label, record.category, record.json_output] for record in records]
        return result
    except Exception as e:
        print(f"Failed to read records: {e}")
        return []
    
def get_json_by_id(record_id):
    """
    Retrieve the JSON data from the database for the given ID and retrun it.
    
    Args:
        record_id (int): The ID of the record in the database.
        
    Returns:
         motion_data (dict): The JSON data as a Python dictionary.
    """
    try:
        # Query the database for the record with the given ID
        record = session.query(VideoData).filter_by(id=record_id).first()
        if not record:
            raise ValueError(f"No record found with ID {record_id}")

        # Load the JSON data from the record
        motion_data = json.loads(record.json_output)

        return motion_data
    
    except Exception as e:
        print(f"Failed to retrieve or convert data for ID {record_id}: {e}")
        return None


def drop_video_data_table():
    """
    Drop the video_data table if it exists.
    """
    if engine is None:
        raise RuntimeError("Database is not initialized. Call create_DB() first.")

    with engine.begin() as connection:  # Use `begin` to handle transactions automatically
        try:
            connection.execute(text("DROP TABLE IF EXISTS video_data CASCADE"))  # Add CASCADE to drop dependencies
            print("video_data table dropped successfully!")
        except Exception as e:
            print(f"Failed to drop video_data table: {e}")


# Create the database tables
def create_db():
    Base.metadata.create_all(engine)
    print("Database and tables created successfully!")

    # Example usage
    # Add a single JSON file
    # add_json_file("motion_data/brother.json", "first_model")

    # Add all JSON files from a folder
    # add_json_files_from_folder("generated_motion_data", "first_model")

if __name__ == "__main__":
    create_db()