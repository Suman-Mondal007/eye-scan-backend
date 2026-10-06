from app.model_loader import model
from app.preprocess import preprocess

def predict_image(image):
    """
    Run cataract prediction on a PIL Image.
    Returns: { prediction, confidence, confidence_pct }
    
    The model returns a raw float 0.0 - 1.0 where:
      - Values > 0.5 = Positive Cataract detected
      - Values <= 0.5 = No Cataract (Normal)
    
    confidence is the RAW model output (0.0 - 1.0).
    We also return confidence_pct which is a human-readable percentage.
    """
    processed = preprocess(image)
    
    prediction = model.predict(processed, verbose=0)
    
    # Raw confidence from model output (0.0 to 1.0)
    raw_confidence = float(prediction[0][0])
    
    is_positive = raw_confidence > 0.5

    # For "Positive Cataract", confidence shown = raw_confidence
    # For "No Cataract", confidence shown = 1 - raw_confidence (confidence in the negative class)
    display_confidence = raw_confidence if is_positive else (1.0 - raw_confidence)
    
    return {
        "prediction": "Positive Cataract" if is_positive else "No Cataract",
        "confidence": raw_confidence,              # raw model output (0-1 float)
        "confidence_pct": round(display_confidence * 100, 2),  # display percentage
        "is_positive": is_positive
    }