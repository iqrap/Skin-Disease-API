from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from model import get_model
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        model = get_model()
        result = model.predict(image_bytes)
        
        descriptions = {
            'Acne': 'Clogged pores, pimples, and oily skin.',
            'Benign Tumors': 'Non-cancerous growths like moles.',
            'Lichen': 'Inflammatory skin condition with purple bumps.',
            'No Disease': 'No significant skin condition detected.',
            'Vitiligo': 'Loss of skin pigment causing white patches.'
        }
        
        result['description'] = descriptions.get(result['diseaseName'], 'Please consult a dermatologist')
        return result
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)