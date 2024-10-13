# main.py

import io
import pickle
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import PIL.Image
import PIL.ImageOps

# Load the trained model
with open('mnist_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict-image/")
async def predict_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        pil_image = PIL.Image.open(io.BytesIO(contents)).convert('L')  # Convert to grayscale
        pil_image = PIL.ImageOps.invert(pil_image)  # Invert colors
        pil_image = pil_image.resize((28, 28), PIL.Image.Resampling.LANCZOS)  # Resize to 28x28
        img_array = np.array(pil_image).reshape(1, -1)  # Flatten the image
        prediction = model.predict(img_array)
        return {"prediction": int(prediction[0])}  # Use lowercase key
    except Exception as e:
        return {"error": str(e)}
