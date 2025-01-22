from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Depends
from sqlalchemy.orm import Session
from pathlib import Path
from database import ProcessedImage, get_db
from process_worker import process_image

app = FastAPI()
UPLOAD_DIR = Path("upload")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Image Processing API!"}

@app.post("/upload/")
async def upload_image(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db)  # Correctly retrieve the database session
):
    file_path = UPLOAD_DIR / file.filename

    # Save the uploaded file
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Create a database entry
    db_entry = ProcessedImage(file_name=file.filename, status="pending")
    db.add(db_entry)
    db.commit()
    db.refresh(db_entry)

    # Add background task for processing
    background_tasks.add_task(process_image, file_path, db_entry.id)

    return {"message": "File uploaded successfully", "id": db_entry.id}

@app.get("/status/{image_id}")
def get_status(
    image_id: int,
    db: Session = Depends(get_db)  # Correctly retrieve the database session
):
    db_entry = db.query(ProcessedImage).filter(ProcessedImage.id == image_id).first()
    if not db_entry:
        raise HTTPException(status_code=404, detail="Image not found")

    return {
        "id": db_entry.id,
        "file_name": db_entry.file_name,
        "status": db_entry.status,
        "result": db_entry.result,
        "error_message": db_entry.error_message
    }