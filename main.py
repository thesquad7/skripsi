import cv2
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime
from io import BytesIO
from PIL import Image
import base64
from ml_operator import get_face_landmarks,draw_landmarks,detect_face_similarity,model

app = FastAPI()

@app.post("/detectingbell")
async def detecting_bell(
    file: UploadFile = File(...),
    name: str = Form(...),
    detect: str = Form(...)
):
    contents = await file.read()
    image = Image.open(BytesIO(contents))

    # Get face landmarks
    face_landmarks, mp_face_landmarks = get_face_landmarks(image)
    if not face_landmarks:
        return JSONResponse(status_code=400, content={"message": "No face detected"})

    # Get similarity percentage
    similarity_percentage = detect_face_similarity(model, face_landmarks)

    # Draw landmarks on the image
    image_with_landmarks = draw_landmarks(image, mp_face_landmarks)

    # Encode image to base64
    _, buffer = cv2.imencode('.png', image_with_landmarks)
    img_str = base64.b64encode(buffer).decode('utf-8')

    return {
        "name": name,
        "detect": detect,
        "similarity_percentage": similarity_percentage,
        "image_with_landmarks": img_str
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
