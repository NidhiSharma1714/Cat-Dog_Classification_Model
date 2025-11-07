import sys
import os
from pathlib import Path

import numpy as np
from PIL import Image

from continuous_learning import get_continuous_learner

# --------------------------
# Platform-dependent import
# --------------------------
try:
    import tflite_runtime.interpreter as tflite
    print("✅ Using tflite-runtime")
except ImportError:
    import tensorflow as tf
    tflite = tf.lite
    print("⚠️ Using full TensorFlow (bigger, slower)")

# --------------------------
# Load TFLite model
# --------------------------
MODEL_PATH = "cat_dog_classifier.tflite"  # Fixed path


def _load_interpreter():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"TFLite model not found: {MODEL_PATH}")
    interpreter_obj = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter_obj.allocate_tensors()
    return interpreter_obj


interpreter = _load_interpreter()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def _reload_interpreter():
    global interpreter, input_details, output_details
    interpreter = _load_interpreter()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()


continuous_learner = get_continuous_learner(on_model_updated=_reload_interpreter)

# --------------------------
# Prediction function
# --------------------------
def predict(image_path):
    # Preprocess
    img = Image.open(image_path).resize((150, 150))
    img_array = np.array(img, dtype=np.float32) / 255.0   # normalize
    img_array = np.expand_dims(img_array, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]  # e.g. [0.92]

    # Sigmoid -> probability of Dog
    prob_dog = float(output_data[0])
    prob_cat = 1.0 - prob_dog

    if prob_dog >= 0.5:
        prediction_class = "Dog"
        confidence = prob_dog * 100
    else:
        prediction_class = "Cat"
        confidence = prob_cat * 100

    # Actual from folder name (optional)
    actual_class = os.path.basename(os.path.dirname(image_path))

    # Print results
    print("Possible predictions: Cat and Dog")
    print(f"Predicted:  {prediction_class}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Probabilities -> Cat: {prob_cat:.2f}, Dog: {prob_dog:.2f}")

    # Continuous learning hook (best-effort)
    try:
        updated = continuous_learner.update_with_image(Path(image_path).resolve(), prediction_class)
        if updated:
            print("[ContinuousLearning] Model refreshed with latest sample.")
    except Exception as exc:
        print(f"[ContinuousLearning] Skipped update: {exc}")

# --------------------------
# Run script from CLI
# --------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)
    predict(image_path)
