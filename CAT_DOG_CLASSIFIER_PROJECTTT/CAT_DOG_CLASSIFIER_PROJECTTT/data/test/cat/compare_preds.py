# compare_preds.py
import sys
import os
import numpy as np
from PIL import Image

# Interpreter selection: tflite_runtime if available, else tensorflow.lite
try:
    import tflite_runtime.interpreter as tflite_rt
    Interpreter = tflite_rt.Interpreter
    print("Using tflite_runtime.interpreter")
except Exception:
    try:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
        print("Using tensorflow.lite.Interpreter (tensorflow installed)")
    except Exception as e:
        raise RuntimeError("Install tensorflow or tflite-runtime in your venv.") from e

MODEL_PATH = "CAT_DOG_CLASSIFIER_PROJECTTT/cat_dog_classifier.tflite"
if not os.path.exists(MODEL_PATH):
    # fallback to repo root model if different
    if os.path.exists("cat_dog_classifier.tflite"):
        MODEL_PATH = "cat_dog_classifier.tflite"
    else:
        raise FileNotFoundError(f"TFLite model not found at expected paths. Checked: {MODEL_PATH} and cat_dog_classifier.tflite")

def load_image(path, size=(150,150)):
    img = Image.open(path).convert("RGB").resize(size)
    arr = np.array(img).astype(np.float32) / 255.0
    return arr

def predict(interpreter, x):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    x_in = np.expand_dims(x, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], x_in)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index'])
    # handle shape differences
    out = out.reshape(-1)
    # if model outputs a single logit, convert to two-class probabilities with sigmoid
    if out.size == 1:
        p1 = 1.0 / (1.0 + np.exp(-out[0]))  # sigmoid -> prob of class 1 (dog)
        probs = np.array([1-p1, p1])
    else:
        # assume softmax-like output for N classes
        exp = np.exp(out - np.max(out))
        probs = exp / np.sum(exp)
    return probs

def compare(path_clean, path_adv):
    if not os.path.exists(path_clean):
        raise FileNotFoundError(f"Clean image not found: {path_clean}")
    if not os.path.exists(path_adv):
        raise FileNotFoundError(f"Adv image not found: {path_adv}")

    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    x_clean = load_image(path_clean)
    x_adv = load_image(path_adv)

    probs_clean = predict(interpreter, x_clean)
    probs_adv = predict(interpreter, x_adv)

    class_names = ["Cat","Dog"]
    pred_clean = class_names[int(np.argmax(probs_clean))]
    pred_adv = class_names[int(np.argmax(probs_adv))]

    l2 = np.linalg.norm((x_clean - x_adv).reshape(-1))
    linf = np.max(np.abs(x_clean - x_adv))

    print("=== CLEAN IMAGE ===")
    print("path:", path_clean)
    print("pred:", pred_clean)
    print("probs -> Cat: {:.4f}, Dog: {:.4f}".format(probs_clean[0], probs_clean[1]))
    print()
    print("=== ADV IMAGE ===")
    print("path:", path_adv)
    print("pred:", pred_adv)
    print("probs -> Cat: {:.4f}, Dog: {:.4f}".format(probs_adv[0], probs_adv[1]))
    print()
    print("=== DISTORTION ===")
    print(f"L2: {l2:.6f}, L-inf: {linf:.6f}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_preds.py <clean_image_path> <adv_image_path>")
        sys.exit(1)
    clean = sys.argv[1]
    adv = sys.argv[2]
    compare(clean, adv)
