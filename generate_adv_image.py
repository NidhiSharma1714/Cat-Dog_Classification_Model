import os
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------------------------------
# 1️⃣  Automatically locate model.tflite
# ---------------------------------------
project_dir = os.path.dirname(os.path.abspath(__file__))

tflite_model_path = None
for root, dirs, files in os.walk(project_dir):
    for file in files:
        if file.endswith(".tflite"):
            tflite_model_path = os.path.join(root, file)
            break
    if tflite_model_path:
        break

if not tflite_model_path:
    raise FileNotFoundError("❌ No .tflite model found in your project folder!")

print(f"✅ Using model: {tflite_model_path}")

# ---------------------------------------
# 2️⃣  Load the TFLite model
# ---------------------------------------
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Detect expected input size dynamically
_, height, width, channels = input_details[0]['shape']
print(f"📏 Model expects input size: {width}x{height}")

# ---------------------------------------
# 3️⃣  Load any clean image (e.g., a Cat)
# ---------------------------------------
image_path = os.path.join(project_dir, "dataset", "train", "Cat", "1890.jpg")
if not os.path.exists(image_path):
    raise FileNotFoundError(f"❌ Image not found at {image_path}")

image = Image.open(image_path).convert("RGB").resize((width, height))
image_np = np.array(image, dtype=np.float32) / 255.0
image_np = np.expand_dims(image_np, axis=0)

# ---------------------------------------
# 4️⃣  Create a simple adversarial noise
# ---------------------------------------
epsilon = 0.05
noise = np.random.normal(0, epsilon, image_np.shape).astype(np.float32)  # ✅ convert to float32
adv_image_np = np.clip(image_np + noise, 0, 1).astype(np.float32)         # ✅ ensure float32

# ---------------------------------------
# 5️⃣  Predict both original & adversarial
# ---------------------------------------
def predict(img_np):
    img_np = img_np.astype(np.float32)  # ✅ ensure float32 before passing
    interpreter.set_tensor(input_details[0]['index'], img_np)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    return output

orig_pred = predict(image_np)
adv_pred = predict(adv_image_np)

# ---------------------------------------
# 6️⃣  Save adversarial image
# ---------------------------------------
os.makedirs("adv_results", exist_ok=True)
adv_image_path = os.path.join("adv_results", "adv_sample.png")
Image.fromarray((adv_image_np[0] * 255).astype(np.uint8)).save(adv_image_path)

# ---------------------------------------
# 7️⃣  Show results
# ---------------------------------------
labels = ["Cat", "Dog"]
print(f"\n✅ Adversarial image saved at: {adv_image_path}")
print(f"🧩 Original Prediction: {labels[int(np.argmax(orig_pred))]} -> {orig_pred}")
print(f"⚠️ Adversarial Prediction: {labels[int(np.argmax(adv_pred))]} -> {adv_pred}")
