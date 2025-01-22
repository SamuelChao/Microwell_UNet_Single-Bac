from sqlalchemy import Column, Integer, String, Text, JSON, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "sqlite:///./test.db"  # Replace with PostgreSQL or other DB if needed

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ProcessedImage(Base):
    __tablename__ = "processed_images"

    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String, unique=True, index=True)
    status = Column(String, default="pending")  # pending, processing, completed, failed
    result = Column(JSON, nullable=True)  # Store detection and mask data
    error_message = Column(Text, nullable=True)

def init_db():
    Base.metadata.create_all(bind=engine)

from sqlalchemy.orm import Session
from typing import Generator

def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()