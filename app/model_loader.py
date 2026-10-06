import os
from tensorflow.keras.models import load_model

# Resolve the model path relative to THIS file, so it works
# regardless of what directory uvicorn is started from
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_THIS_DIR, '..', 'models', 'cataract_model.h5')
_MODEL_PATH = os.path.normpath(_MODEL_PATH)

print(f"[OphthalmoScan] Loading trained model from: {_MODEL_PATH}")
model = load_model(_MODEL_PATH)
print("[OphthalmoScan] Model loaded successfully.")