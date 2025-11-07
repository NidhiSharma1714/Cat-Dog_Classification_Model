import sys
import numpy as np
from PIL import Image
import tensorflow as tf

# --------------------------
# Load TFLite model
# --------------------------
MODEL_PATH = "cat_dog_classifier.tflite"

def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

def predict_image(interpreter, input_details, output_details, image_path):
    img = Image.open(image_path).convert('RGB').resize((150, 150))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]

    pred_class = "Dog" if output[0] > 0.5 else "Cat"
    confidence = float(output[0]) if output[0] > 0.5 else 1 - float(output[0])
    return pred_class, confidence, output

def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_preds.py <clean_image_path> <adv_image_path>")
        sys.exit(1)

    clean_path, adv_path = sys.argv[1], sys.argv[2]

    interpreter, input_details, output_details = load_tflite_model(MODEL_PATH)

    clean_pred, clean_conf, clean_probs = predict_image(interpreter, input_details, output_details, clean_path)
    adv_pred, adv_conf, adv_probs = predict_image(interpreter, input_details, output_details, adv_path)

    print("\n🧩 Model Comparison Results:")
    print(f"Clean image ({clean_path}) → {clean_pred} ({clean_conf*100:.2f}%)")
    print(f"Adversarial image ({adv_path}) → {adv_pred} ({adv_conf*100:.2f}%)")

    if clean_pred != adv_pred:
        print("\n⚠️ Model prediction CHANGED due to adversarial attack!")
    else:
        print("\n✅ Model prediction remained the SAME.")

if __name__ == "__main__":
    main()
