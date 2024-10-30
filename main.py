from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model # type: ignore
from PIL import Image, ImageOps
from scipy.stats import entropy
import numpy as np
import io
import time
from tensorflow.keras.layers import DepthwiseConv2D  # type: ignore


# Initialiser l'application FastAPI
app = FastAPI(
    title="Dental Diseases Prediction API",
    description="Cette API prédit le type de maladie bucco-dentaire à partir d'images.",
    version="1.0.0"
)


# Route pour accêder a la page d'accueil
@app.get("/")
def read_root():
    return {"message": "API Dental Diseases Prediction"}

def custom_depthwise_conv2d(*args, **kwargs):
    if 'groups' in kwargs:
        del kwargs['groups']  # Retirer 'groups'
    return DepthwiseConv2D(*args, **kwargs)

# Charger le modèle
model = load_model("Model3.keras", compile=False)

# Charger les étiquettes
with open("labels.txt", "r") as file:
    class_names = [line.strip() for line in file.readlines()]

# Fonction pour prétraiter l'image
def preprocess(image):
    # Redimensionner l'image à 224x224
    image = ImageOps.fit(image, (180, 180), Image.Resampling.LANCZOS)
    
    # Convertir l'image en tableau numpy et s'assurer qu'elle a 3 canaux
    image_array = np.asarray(image.convert("RGB"), dtype=np.float32)
    
    # Normaliser l'image
    normalized_image_array = (image_array / 127.5) - 1
    
    # Créer le tableau de données pour le modèle
    data = np.ndarray(shape=(1, 180, 180, 3), dtype=np.float32)
    data[0] = normalized_image_array

    return data

# Endpoint pour les prédictions
@app.post("/predict/", summary="Upload an image for dental disease prediction", description="This endpoint allows you to upload an image of the teeth and get a prediction of the dental disease.")
async def predict(file: UploadFile = File(...)):
    """
    Charge une image et la passe au modèle pour prédire.
    
    - **file**: fichier image (jpeg ou png) à analyser
    """
    # Lire l'image envoyée
    try:
        image = Image.open(io.BytesIO(await file.read()))
    except Exception as e:
        return {"error": f"Invalid image file: {str(e)}"}
    
    # Pré-traiter l'image
    processed_image = preprocess(image)
    
    # Calculer le temps de traitement
    start_time = time.time()
    
    # Faire la prédiction avec le modèle
    prediction = model.predict(processed_image)
    processing_time = time.time() - start_time

    # Trouver la classe avec la probabilité la plus élevée
    predicted_class_index = np.argmax(prediction[0])
    predicted_class = class_names[predicted_class_index]
    
    # Calculer la marge de confiance
    sorted_probs = np.sort(prediction[0])[::-1]
    confidence_margin = sorted_probs[0] - sorted_probs[1]
    
    # Calculer l'entropie
    uncertainty = entropy(prediction[0])  # calcul de l'entropie
    
    # Retourner la prédiction sous forme JSON
    return {
        "prediction": predicted_class,
        "confidence": float(prediction[0][predicted_class_index]),
        "confidence_margin": float(confidence_margin),
        "uncertainty": float(uncertainty),
        "processing_time": float(processing_time)
    }

# Commande pour démarrer le serveur FastAPI
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
