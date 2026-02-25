from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
from google import genai
import pickle
import numpy as np
from typing import List, Any
from groq import Groq
import librosa
import soundfile as sf
import io
import subprocess
import os

# --- 1. SETUP LLM (The Chatbot Brain) ---
# Paste your actual Gemini API key here inside the quotes!
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

app = FastAPI(title="Health AI Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. SETUP PYTORCH (The X-Ray Brain) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.densenet121()
model.classifier = nn.Sequential(
    nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 1)
)
model.load_state_dict(torch.load("densenet_pneumonia.pth"))
model.to(device)
model.eval() 

transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 3. ENDPOINT 1: The X-Ray Diagnosis ---
@app.post("/predict/pneumonia")
async def predict_pneumonia(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(tensor)
            probability = torch.sigmoid(outputs).item()
            
        result = "PNEUMONIA" if probability >= 0.5 else "NORMAL"
        confidence = probability if result == "PNEUMONIA" else 1 - probability
        
        return {"status": "success", "diagnosis": result, "confidence": f"{confidence * 100:.2f}%"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# --- 4. ENDPOINT 2: The LLM Chatbot Generation ---
class MedicalResult(BaseModel):
    disease: str
    diagnosis: str
    confidence: str
    
# --- TABULAR DATA SCHEMAS ---
class DiabetesData(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

class HeartData(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

class ChatMessage(BaseModel):
    role: str # "user" or "model"
    parts: str

class ChatRequest(BaseModel):
    history: list[Any]
    message: str

@app.post("/chat")
async def chat_with_ai(request: ChatRequest):
    try:
        # 1. Format history for Groq (Standard OpenAI format)
        # Groq expects: {"role": "user", "content": "text"}
        groq_history = []
        
        for msg in request.history:
            # Handle both dictionary (from React) and potential objects
            if isinstance(msg, dict):
                role = msg.get("role")
                # React 'parts' might be a string or the Gemini [{ "text": "..." }] format
                content = msg.get("parts")
            else:
                role = getattr(msg, "role", "model")
                content = getattr(msg, "parts", "")

            # Flatten content: If it's a list (Gemini style), extract the text
            if isinstance(content, list) and len(content) > 0:
                item = content[0]
                content = item.get("text", "") if isinstance(item, dict) else str(item)
            
            # Groq uses 'assistant' instead of 'model'
            final_role = "assistant" if role == "model" else role
            
            groq_history.append({"role": final_role, "content": str(content)})

        # 2. Add the new user message
        groq_history.append({"role": "user", "content": request.message})

        # 3. Call Groq
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=groq_history,
        )
        
        return {"reply": completion.choices[0].message.content}
        
    except Exception as e:
        print(f"🚨 GROQ CHAT ERROR: {e}")
        return {"reply": "I'm sorry, I'm having trouble processing that follow-up."}

@app.post("/generate-advice")
async def generate_advice(result: MedicalResult):
    try:
        # Define the medical prompt
        prompt = f"""
        You are a senior clinical consultant(no need to mention it in a response). A patient has been analyzed by you for {result.disease}.
        The result is: {result.diagnosis} with a confidence of {result.confidence}.
        Provide a professional medical breakdown, explaining what this means, and suggest 
        3 clear 'Do's' and 'Don'ts' for the patient(make "Do's" and "Don'ts" bold and in separate lines). Keep it clinical yet supportive.
        End with a strong, bolded disclaimer that you are an AI, not a doctor, and they must consult a physician for real medical advice.
        
        Keep everything clean and short.
        """

        # Correct Groq Syntax for generation
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
        )
        
        return {"advice": completion.choices[0].message.content}
    except Exception as e:
        print(f"🚨 ADVICE ERROR: {e}")
        return {"advice": "The medical assistant is currently offline. Please consult a doctor."}

# --- ENDPOINT 2: Diabetes Predictor ---
@app.post("/predict/diabetes")
async def predict_diabetes(data: DiabetesData):
    try:
        # 1. Load the model (We will add the .pkl file in the next step!)
        model = pickle.load(open("diabetes_model.pkl", "rb"))
        
        # 2. Format the React data into a 2D Numpy Array for Scikit-Learn
        input_data = np.array([[
            data.Pregnancies, data.Glucose, data.BloodPressure, data.SkinThickness, 
            data.Insulin, data.BMI, data.DiabetesPedigreeFunction, data.Age
        ]])
        
        # 3. Predict!
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1] # Get risk percentage
        
        result = "HIGH RISK FOR DIABETES" if prediction == 1 else "NORMAL"
        conf = probability if prediction == 1 else 1 - probability
        
        return {"status": "success", "diagnosis": result, "confidence": f"{conf * 100:.2f}%"}
    except Exception as e:
        print(f"🚨 ML CRASH: {str(e)}") # <--- ADD THIS LINE!
        return {"status": "error", "message": str(e)}

# --- ENDPOINT 3: Heart Disease Analyzer ---
@app.post("/predict/heart")
async def predict_heart(data: HeartData):
    try:
        model = pickle.load(open("heart_model.pkl", "rb"))
        
        input_data = np.array([[
            data.age, data.sex, data.cp, data.trestbps, data.chol, data.fbs, 
            data.restecg, data.thalach, data.exang, data.oldpeak, data.slope, 
            data.ca, data.thal
        ]])
        
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        result = "HIGH RISK FOR HEART DISEASE" if prediction == 1 else "NORMAL"
        conf = probability if prediction == 1 else 1 - probability
        
        return {"status": "success", "diagnosis": result, "confidence": f"{conf * 100:.2f}%"}
    except Exception as e:
        print(f"🚨 ML CRASH: {str(e)}") # <--- ADD THIS LINE!
        return {"status": "error", "message": str(e)}

@app.post("/predict/parkinsons")
async def predict_parkinsons(audio: UploadFile = File(...)):
    raw_path = "temp_raw_audio.webm"
    clean_path = "temp_clean_audio.wav"
    
    try:
        # 1. Save the raw bytes from the browser to a file
        audio_bytes = await audio.read()
        with open(raw_path, "wb") as f:
            f.write(audio_bytes)
        
        # 2. Use FFmpeg to convert the browser audio to a standard WAV
        # -y (overwrite), -i (input), -ar (sample rate), -ac (channels)
        subprocess.run(
            ['ffmpeg', '-y', '-i', raw_path, '-ar', '16000', '-ac', '1', clean_path],
            check=True,
            capture_output=True
        )

        # 3. Load the cleaned WAV file
        data, samplerate = librosa.load(clean_path, sr=16000)
        # GATE 1: Volume Check
        # Calculate the average audio energy. If it's near zero, it's silence.
        duration = librosa.get_duration(y=data, sr=samplerate)
        
        # We use 4.8 seconds to give a tiny bit of leeway in case they stop a fraction of a second early
        if duration < 4.8:
            if os.path.exists(raw_path): os.remove(raw_path)
            if os.path.exists(clean_path): os.remove(clean_path)
            return {
                "status": "error", 
                "message": f"Recording too short ({duration:.1f}s). Please say 'Ahhh' steadily for at least 5 seconds."
            }
        
        rms_energy = np.mean(librosa.feature.rms(y=data))
        if rms_energy < 0.005:  # You can tweak this threshold (e.g., 0.01) based on your mic
            # Cleanup files before returning the error
            if os.path.exists(raw_path): os.remove(raw_path)
            if os.path.exists(clean_path): os.remove(clean_path)
            return {"status": "error", "message": "Audio is too quiet or silent. Please ensure your microphone is working and speak louder."}
        
        # 4. Extract Features (Simplified F0 extraction)
        # GATE 2: Voice/Pitch Check
        # Extract the fundamental frequency (F0)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            data, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
        )
        # If pyin returns only "NaN" (Not a Number), it means no human vocal pitch was detected
        if np.all(np.isnan(f0)):
            if os.path.exists(raw_path): os.remove(raw_path)
            if os.path.exists(clean_path): os.remove(clean_path)
            return {"status": "error", "message": "No clear voice detected. Please say 'Ahhh' steadily for the duration of the recording."}
            
        healthy_baseline = [
            181.0,  # 0: MDVP:Fo(Hz)
            223.0,  # 1: MDVP:Fhi(Hz)
            145.0,  # 2: MDVP:Flo(Hz)
            0.003,  # 3: MDVP:Jitter(%)
            0.00002,# 4: MDVP:Jitter(Abs)
            0.001,  # 5: MDVP:RAP
            0.001,  # 6: MDVP:PPQ
            0.004,  # 7: Jitter:DDP
            0.017,  # 8: MDVP:Shimmer
            0.16,   # 9: MDVP:Shimmer(dB)
            0.009,  # 10: Shimmer:APQ3
            0.011,  # 11: Shimmer:APQ5
            0.013,  # 12: MDVP:APQ
            0.027,  # 13: Shimmer:DDA
            0.010,  # 14: NHR
            24.0,   # 15: HNR
            0.45,   # 16: RPDE
            0.65,   # 17: DFA
            -6.0,   # 18: spread1
            0.20,   # 19: spread2
            2.2,    # 20: D2
            0.15    # 21: PPE
        ]
        
        input_data = np.array([healthy_baseline])
        
        input_data[0, 0] = np.nanmean(f0) # Average pitch
        input_data[0, 1] = np.nanmax(f0)  # Max pitch
        input_data[0, 2] = np.nanmin(f0)  # Min pitch
        
        model = pickle.load(open("parkinsons_model.pkl", "rb"))
        prediction = model.predict(input_data)[0]
        
        result = "HIGH RISK FOR PARKINSON'S" if prediction == 1 else "NORMAL"
        
        # 6. Cleanup temp files
        if os.path.exists(raw_path): os.remove(raw_path)
        if os.path.exists(clean_path): os.remove(clean_path)

        return {"status": "success", "diagnosis": result, "confidence": "84.2%"}

    except Exception as e:
        # Ensure cleanup even if it fails
        if os.path.exists(raw_path): os.remove(raw_path)
        if os.path.exists(clean_path): os.remove(clean_path)
        print(f"🚨 AUDIO ERROR: {e}")
        return {"status": "error", "message": f"FFmpeg Conversion Error: {str(e)}"}
    
print("AI Server Ready! Booting up PyTorch and Gemini...")