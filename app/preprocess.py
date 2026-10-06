import numpy as np

def preprocess(image):
    # Force RGB mode (handles RGBA, Grayscale, Palette images from web browser uploads)
    image = image.convert('RGB')
    
    # Resize to model input size (224x224)
    image = image.resize((224, 224))
    
    # Normalize to [0, 1]
    image = np.array(image, dtype=np.float32) / 255.0
    
    # Add batch dimension -> shape (1, 224, 224, 3)
    image = image.reshape(1, 224, 224, 3)
    
    return image