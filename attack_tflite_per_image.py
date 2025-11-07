# attack_tflite_per_image.py
import os
import numpy as np
from PIL import Image
from tflite_predict_wrapper import TFLiteWrapper
from art.estimators.classification import BlackBoxClassifier
from art.attacks.evasion import HopSkipJump

TFLITE_MODEL = "cat_dog_classifier.tflite"
DATA_ROOT = r"C:\Users\admin\Downloads\dataSet\PetImages"  # or leave auto-detect logic as before
CAT_FOLDER = os.path.join(DATA_ROOT, "Cat")
DOG_FOLDER = os.path.join(DATA_ROOT, "Dog")
MAX_PER_CLASS = 6
HSJ_MAX_ITER = 10
HSJ_MAX_EVAL = 500
HSJ_INIT_EVAL = 50
SAVE_DIR = "adv_outputs_per_image"
NB_CLASSES = 2

def load_images(folder, max_samples, target_hw):
    imgs, fnames = [], []
    H,W,C = target_hw
    for f in sorted(os.listdir(folder))[:max_samples]:
        p = os.path.join(folder, f)
        try:
            with Image.open(p) as im:
                im = im.convert("RGB").resize((W,H))
                arr = np.array(im).astype("float32") / 255.0
                imgs.append(arr); fnames.append(f)
        except Exception:
            continue
    return np.array(imgs), fnames

def save_side_by_side(orig, adv, out_path):
    o = (np.clip(orig,0,1)*255).astype('uint8'); a = (np.clip(adv,0,1)*255).astype('uint8')
    H,W,_ = o.shape
    canvas = Image.new("RGB",(W*2,H))
    canvas.paste(Image.fromarray(o),(0,0)); canvas.paste(Image.fromarray(a),(W,0))
    canvas.save(out_path)

def main():
    wrapper = TFLiteWrapper(TFLITE_MODEL)
    target_hw = getattr(wrapper, "model_input_hw", (150,150,3))

    Xc, fn_c = load_images(CAT_FOLDER, MAX_PER_CLASS, target_hw)
    Xd, fn_d = load_images(DOG_FOLDER, MAX_PER_CLASS, target_hw)
    X = np.concatenate([Xc, Xd], axis=0)
    y = np.concatenate([np.zeros(len(Xc),dtype=int), np.ones(len(Xd),dtype=int)], axis=0)
    fnames = fn_c + fn_d
    os.makedirs(SAVE_DIR, exist_ok=True)

    classifier = BlackBoxClassifier(predict_fn=wrapper.predict, input_shape=target_hw, nb_classes=NB_CLASSES)

    # Attack images one-by-one so we can save and inspect immediately
    attack = HopSkipJump(classifier=classifier, max_iter=HSJ_MAX_ITER, max_eval=HSJ_MAX_EVAL, init_eval=HSJ_INIT_EVAL, init_size=5, verbose=False)

    for i in range(len(X)):
        x = np.expand_dims(X[i], axis=0)
        label = y[i]
        print(f"Attacking sample {i+1}/{len(X)}: {fnames[i]} (true label={label})")
        try:
            x_adv = attack.generate(x)
        except Exception as e:
            print(" Attack failed for sample", fnames[i], ":", e)
            continue

        probs_clean = classifier.predict(x)
        probs_adv = classifier.predict(x_adv)
        pred_clean = int(np.argmax(probs_clean, axis=1)[0])
        pred_adv = int(np.argmax(probs_adv, axis=1)[0])
        print(f"  clean pred={pred_clean}, adv pred={pred_adv}, clean probs={probs_clean[0]}, adv probs={probs_adv[0]}")

        out_name = os.path.splitext(fnames[i])[0] + f"_idx{i}_c{pred_clean}_a{pred_adv}.png"
        save_side_by_side(x[0], x_adv[0], os.path.join(SAVE_DIR, out_name))
        print("  saved:", out_name)

if __name__ == '__main__':
    main()
