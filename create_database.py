import os
import json
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import declarative_base, sessionmaker

# Database connection configuration
DATABASE_URL = "postgresql://postgres:lenos4022499@localhost:5432/postgres"
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

# Function to extract label up to the first underscore
def extract_label(file_name):
    return file_name.split('_')[0]

# Function to add a single JSON file to the database
def add_json_file(file_path, category):
    try:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        label = extract_label(file_name)
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


# Create the database tables
if __name__ == "__main__":
    Base.metadata.create_all(engine)
    print("Database and tables created successfully!")

    # Example usage
    # Add a single JSON file
    # add_json_file("motion_data/brother.json", "first_model")

    # Add all JSON files from a folder
    add_json_files_from_folder("generated_motion_data", "first_model")