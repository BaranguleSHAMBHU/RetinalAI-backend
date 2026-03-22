from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image

app = FastAPI(title="RetinalAI Disease Classifier API")

# Allow your Next.js frontend (running on port 3000) to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Load the trained model (Ensure the filename matches exactly!)
print("Loading model... this might take a few seconds.")
model = load_model('best_retinal_model.keras')
print("Model loaded successfully!")

# 2. Define the classes EXACTLY as they appeared in your training dataset folders
CLASS_NAMES = ['Cataract', 'Diabetic_Retinopathy', 'Glaucoma', 'Normal']

@app.get("/")
def read_root():
    return {"message": "RetinalAI Backend is running!"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        # Read the uploaded image file
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Preprocess the image to match the DenseNet169 training input
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize pixel values
        
        # Make the prediction
        predictions = model.predict(img_array)[0]
        
        # Format the results
        results_dict = {CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))}
        
        # Sort by highest confidence
        sorted_results = sorted(results_dict.items(), key=lambda item: item[1], reverse=True)
        top_disease = sorted_results[0][0]
        top_confidence = sorted_results[0][1]

        # Clean up 'Diabetic_Retinopathy' to look nice on the frontend
        top_disease_clean = top_disease.replace('_', ' ')
        formatted_breakdown = [[k.replace('_', ' '), v] for k, v in sorted_results]

        return {
            "disease": top_disease_clean,
            "confidence": top_confidence,
            "breakdown": formatted_breakdown
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))