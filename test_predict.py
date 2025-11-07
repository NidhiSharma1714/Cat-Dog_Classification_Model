# test_predict.py
import numpy as np
from PIL import Image
from tflite_predict_wrapper import TFLiteWrapper

# Path to your tflite
MODEL_PATH = "cat_dog_classifier.tflite"

# Example image path (change to any real image in your dataset)
IMG_PATH = r"C:\Users\admin\Downloads\dataSet\PetImages\Dog\994.jpg"

def load_and_prepare(img_path, size):
    img = Image.open(img_path).convert("RGB").resize((size[1], size[0]))
    arr = np.array(img).astype('float32') / 255.0
    return arr

def main():
    wrapper = TFLiteWrapper(MODEL_PATH)
    H, W, C = wrapper.model_input_hw

    img = load_and_prepare(IMG_PATH, (H, W, C))
    batch = np.expand_dims(img, axis=0)  # shape (1,H,W,C)

    probs = wrapper.predict(batch)  # shape (1, nb_classes)
    print("probs:", probs, "shape:", probs.shape)

    pred_idx = int(np.argmax(probs, axis=1)[0])
    print("predicted class index:", pred_idx)
    # Convention: [class0, class1] -> 0 = cat, 1 = dog (adjust if your mapping is opposite)
    print(f"prob class0 (cat): {probs[0,0]:.6f}, prob class1 (dog): {probs[0,1]:.6f}")

if __name__ == "__main__":
    main()
