import cv2
import json
from pathlib import Path
from sqlalchemy.orm import Session
from database import ProcessedImage, SessionLocal
from detection import detect_circles
from segmentation import predict_mask

def process_image(image_path: Path, image_id: int):
    db = SessionLocal()
    try:
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError("Invalid image file")

        # Update status to processing
        db_entry = db.query(ProcessedImage).filter(ProcessedImage.id == image_id).first()
        db_entry.status = "processing"
        db.commit()

        # Detect circles
        circles = detect_circles(image)
        results = {"circles": [], "masks": []}

        for x, y, r, top_left, bottom_right in circles:
            slice_image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            mask = predict_mask(slice_image)
            results["circles"].append({
                "center": (x, y),
                "radius": r,
                "bounding_box": [top_left, bottom_right]
            })
            results["masks"].append(mask.tolist())

        # Update database entry
        db_entry.status = "completed"
        db_entry.result = results
        db.commit()
    except Exception as e:
        # Update database with error
        db_entry.status = "failed"
        db_entry.error_message = str(e)
        db.commit()
    finally:
        db.close()