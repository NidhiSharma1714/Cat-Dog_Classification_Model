# evaluate_clean_accuracy.py
import os
import numpy as np
from PIL import Image

# Try tflite_runtime first, then tf.lite
try:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
    print("Using tflite_runtime.interpreter")
except Exception as e:
    try:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
        print("Using tensorflow.lite.Interpreter (tensorflow installed)")
    except Exception as e2:
        raise RuntimeError("Neither tflite_runtime nor tensorflow is available. Install tensorflow or tflite-runtime in your venv.") from e2

# Load model
MODEL_PATH = "cat_dog_classifier.tflite"
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess(img_path, target_size=(150,150)):
    img = Image.open(img_path).convert('RGB').resize(target_size)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict(img_path):
    img = preprocess(img_path)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    return output

def main():
    test_dir = "CAT_DOG_CLASSIFIER_PROJECTTT/data/test" if os.path.isdir("CAT_DOG_CLASSIFIER_PROJECTTT/data/test") else "data/test"
    class_names = ["cat", "dog"]
    correct = 0
    total = 0

    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}. Make sure test images exist in data/test/cat and data/test/dog")

    for label in class_names:
        folder = os.path.join(test_dir, label)
        if not os.path.isdir(folder):
            print(f"Warning: folder missing: {folder} -> skipping.")
            continue
        for file in os.listdir(folder):
            if file.lower().endswith(('.jpg','.jpeg','.png')):
                path = os.path.join(folder, file)
                preds = predict(path)
                pred_class = class_names[int(np.argmax(preds))]
                if pred_class == label:
                    correct += 1
                total += 1

    if total == 0:
        print("No test images found. Put images under data/test/cat and data/test/dog.")
    else:
        print(f"✅ Clean accuracy: {correct}/{total} = {correct/total*100:.2f}%")

if __name__ == "__main__":
    main()
