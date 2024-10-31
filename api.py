from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model  # type: ignore
from PIL import Image
from scipy.stats import entropy
import numpy as np
import io
import time
from tensorflow.keras.layers import DepthwiseConv2D # type: ignore

# Initialiser l'application FastAPI
app = FastAPI(
    title="Dental Diseases Prediction API",
    description="Cette API prédit le type de maladies bucco-dentaire à partir d'images couleurs.",
    version="1.0.0"
)

# Définir la couche sans le paramètre 'groups'
def custom_depthwise_conv2d(*args, **kwargs):
    if 'groups' in kwargs:
        del kwargs['groups']  # Retirer 'groups'
    return DepthwiseConv2D(*args, **kwargs)

# Charger le modèle ML (assurez-vous d'ajuster le chemin si nécessaire)
try:
    model = load_model('dental_model.h5', custom_objects={'DepthwiseConv2D': custom_depthwise_conv2d}, compile=False)
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement du modèle: {e}")

# Fonction pour pré-traiter l'image avant de la passer au modèle
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")  # Assurez-vous que l'image a 3 canaux
    image = image.resize((224, 224))  # Ajustez la taille selon votre modèle
    image = np.array(image) / 255.0    # Normalisation
    image = np.expand_dims(image, axis=0)  # Ajouter une dimension batch
    return image

# Route pour accèder a la page d'accueil
@app.get("/")
def read_root():
    return {"message": "API Dental Diseases Prediction"}

# Endpoint pour les prédictions
@app.post("/predict/", summary="Upload an image for dental diseases prediction", 
          description="This endpoint allows you to upload an image of the teeth and get a prediction.")
async def predict(file: UploadFile = File(...)):
    """
    Charge une image de 180x180 pixels et la passe au modèle pour prédire.
    
    - **file**: fichier image (jpeg ou png) à analyser
    """
    # Lire l'image envoyée
    try:
        image = Image.open(io.BytesIO(await file.read()))
    except Exception as e:
        return {"error": "Invalid image file"}
    
    # Pré-traiter l'image
    processed_image = preprocess_image(image)
    
    # Mesurer le temps de traitement
    start_time = time.time()
    
    # Faire la prédiction avec le modèle
    try:
        prediction = model.predict(processed_image)
    except Exception as e:
        return {"error": f"Erreur lors de la prédiction: {e}"}
    
    # Liste des classes
    classes = ["Dental Caries", "Calculus", "Gingivitis", "Hypodontia", "Tooth Discoloration"]
    
    # Trouver la classe avec la probabilité la plus élevée
    predicted_class_index = np.argmax(prediction[0])
    predicted_class = classes[predicted_class_index]
    
    # Calculer la marge de confiance
    sorted_probs = np.sort(prediction[0])[::-1]
    confidence_margin = sorted_probs[0] - sorted_probs[1]
    
    # Calculer l'entropie
    uncertainty = entropy(prediction[0])  # calcul de l'entropie

    # Calculer le temps de traitement
    processing_time = time.time() - start_time

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
    uvicorn.run(app, host="0.0.0.0", port=8090)
