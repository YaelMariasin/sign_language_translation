from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Database connection configuration
DATABASE_URL = "postgresql://username:password@localhost:5432/your_database_name"
engine = create_engine(DATABASE_URL)
Base = declarative_base()

# Define the VideoData table
class VideoData(Base):
    __tablename__ = 'video_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)  # Unique identifier
    video_path = Column(String(255), nullable=False)  # Path to the video file
    transcription = Column(Text, nullable=True)  # Transcription of the video
    gesture_data = Column(Text, nullable=True)  # Information about gestures (e.g., in JSON format)
    category = Column(String(100), nullable=True)  # Category of the video (e.g., "educational", "tutorial")
    created_at = Column(DateTime, default=datetime.utcnow)  # Timestamp when the record is created

# Create a database session
Session = sessionmaker(bind=engine)
session = Session()

# Function to add a new video record
def add_video(video_path, transcription=None, gesture_data=None, category=None):
    try:
        new_video = VideoData(
            video_path=video_path,
            transcription=transcription,
            gesture_data=gesture_data,
            category=category
        )
        session.add(new_video)
        session.commit()
        print(f"Video added successfully with ID: {new_video.id}")
    except Exception as e:
        session.rollback()
        print(f"Failed to add video: {e}")

# Function to remove a video record by ID
def remove_video(video_id):
    try:
        video_to_delete = session.query(VideoData).filter_by(id=video_id).first()
        if video_to_delete:
            session.delete(video_to_delete)
            session.commit()
            print(f"Video with ID {video_id} removed successfully!")
        else:
            print(f"No video found with ID {video_id}")
    except Exception as e:
        session.rollback()
        print(f"Failed to remove video: {e}")

# Create the database tables
if __name__ == "__main__":
    Base.metadata.create_all(engine)
    print("Database and tables created successfully!")

    # Example usage
    add_video(
        video_path="path/to/video.mp4",
        transcription="This is a sample transcription",
        gesture_data='{"gesture": "wave", "hand": "right"}',
        category="educational"
    )
    
    # Example of removing a video
    remove_video(video_id=1)
