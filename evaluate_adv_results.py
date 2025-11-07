# evaluate_adv_results.py
import os
import numpy as np
from PIL import Image
from tflite_predict_wrapper import TFLiteWrapper
import csv

MODEL_PATH = "cat_dog_classifier.tflite"
ADV_DIR = "adv_outputs_per_image"
REPORT_CSV = "adv_report.csv"

def load_img(path, size):
    img = Image.open(path).convert("RGB").resize((size[1], size[0]))
    arr = np.array(img).astype("float32") / 255.0
    return arr

def main():
    wrapper = TFLiteWrapper(MODEL_PATH)
    H,W,C = wrapper.model_input_hw
    rows = [["filename","true_label","clean_pred","adv_pred","clean_prob0","clean_prob1","adv_prob0","adv_prob1","l2","linf"]]

    for fname in sorted(os.listdir(ADV_DIR)):
        if not fname.lower().endswith((".png",".jpg",".jpeg")): continue
        path = os.path.join(ADV_DIR, fname)
        try:
            # file pattern is originalname_idxX_c{clean}_a{adv}.png from per-image script
            parts = fname.rsplit("_", 3)
            orig_name = parts[0] + ".jpg"  # best-effort; you may want to adjust depending on naming
        except Exception:
            orig_name = None

        # open side-by-side and split
        im = Image.open(path).convert("RGB")
        w,h = im.size
        half = w//2
        orig = np.array(im.crop((0,0,half,h))).astype("float32")/255.0
        adv = np.array(im.crop((half,0,w,h))).astype("float32")/255.0

        clean_prob = wrapper.predict(np.expand_dims(orig,axis=0))[0]
        adv_prob = wrapper.predict(np.expand_dims(adv,axis=0))[0]
        clean_pred = int(np.argmax(clean_prob))
        adv_pred = int(np.argmax(adv_prob))
        l2 = np.linalg.norm((adv - orig).ravel(), ord=2)
        linf = np.max(np.abs(adv - orig))

        row = [fname, orig_name or "?", clean_pred, adv_pred,
               float(clean_prob[0]), float(clean_prob[1]),
               float(adv_prob[0]), float(adv_prob[1]),
               float(l2), float(linf)]
        rows.append(row)

    # save CSV
    with open(REPORT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print("Saved:", REPORT_CSV)

if __name__ == "__main__":
    main()
