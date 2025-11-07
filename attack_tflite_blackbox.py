# attack_tflite_blackbox.py
"""
Black-box attack script using ART's HopSkipJump attack against your TFLite model.

Usage:
    python attack_tflite_blackbox.py

Defaults are conservative so it runs quickly for testing. Increase MAX_PER_CLASS,
HSJ_MAX_ITER, HSJ_MAX_EVAL, and HSJ_INIT_EVAL for stronger (but slower) attacks.
"""

import os
import numpy as np
from PIL import Image
from tflite_predict_wrapper import TFLiteWrapper
from art.estimators.classification import BlackBoxClassifier
from art.attacks.evasion import HopSkipJump

# -------------------- CONFIG --------------------
TFLITE_MODEL = "cat_dog_classifier.tflite"

CANDIDATE_DATA_ROOTS = [
    r"C:\Users\admin\Downloads\dataSet\PetImages",
    "dataset/PetImages",
    "data/PetImages",
    "PetImages",
    "dataset"
]

CAT_NAMES = ["Cat", "cat", "Cats", "cats"]
DOG_NAMES = ["Dog", "dog", "Dogs", "dogs"]

MAX_PER_CLASS = 6        # small for quick tests (increase for better statistics)
HSJ_MAX_ITER = 20        # HopSkipJump iterations (increase for stronger attacks)
HSJ_MAX_EVAL = 1000      # per-iteration evaluation budget
HSJ_INIT_EVAL = 100      # initial evaluations to find starting point
NB_CLASSES = 2
SAVE_DIR = "adv_outputs"
# ------------------------------------------------

def find_dataset_root():
    for cand in CANDIDATE_DATA_ROOTS:
        if os.path.exists(cand):
            return cand
    cwd = os.getcwd()
    for entry in os.listdir(cwd):
        p = os.path.join(cwd, entry)
        if os.path.isdir(p):
            children = set(os.listdir(p))
            if any(n in children for n in CAT_NAMES) and any(n in children for n in DOG_NAMES):
                return p
    return None

def find_class_folders(data_root):
    cats = None
    dogs = None
    for name in CAT_NAMES:
        p = os.path.join(data_root, name)
        if os.path.isdir(p):
            cats = p
            break
    for name in DOG_NAMES:
        p = os.path.join(data_root, name)
        if os.path.isdir(p):
            dogs = p
            break
    return cats, dogs

def load_images(folder, max_samples, target_hw):
    imgs = []
    fnames = []
    H, W, C = target_hw
    for fname in sorted(os.listdir(folder)):
        if len(imgs) >= max_samples:
            break
        path = os.path.join(folder, fname)
        try:
            with Image.open(path) as im:
                im = im.convert("RGB").resize((W, H))
                arr = np.array(im).astype("float32") / 255.0
                imgs.append(arr)
                fnames.append(fname)
        except Exception:
            continue
    return np.array(imgs), fnames

def save_side_by_side(orig, adv, out_path):
    o = (np.clip(orig, 0.0, 1.0) * 255).astype(np.uint8)
    a = (np.clip(adv, 0.0, 1.0) * 255).astype(np.uint8)
    H, W, _ = o.shape
    canvas = Image.new("RGB", (W * 2, H))
    canvas.paste(Image.fromarray(o), (0, 0))
    canvas.paste(Image.fromarray(a), (W, 0))
    canvas.save(out_path)

def main():
    print("Loading TFLite wrapper and model...")
    wrapper = TFLiteWrapper(TFLITE_MODEL)

    # Try to infer model input shape (H,W,C)
    try:
        input_shape = wrapper.input_details[0]["shape"]
        target_hw = (int(input_shape[1]), int(input_shape[2]), int(input_shape[3]))
    except Exception:
        # fallback to wrapper attribute if present
        target_hw = getattr(wrapper, "model_input_hw", (150, 150, 3))
    print("Model input shape (H,W,C):", target_hw)

    # find dataset
    data_root = find_dataset_root()
    if not data_root:
        print("Could not auto-detect dataset root. Edit CANDIDATE_DATA_ROOTS or set path manually.")
        return
    print("Dataset root found:", data_root)

    cat_folder, dog_folder = find_class_folders(data_root)
    if not cat_folder or not dog_folder:
        print("Could not find Cat/Dog folders under dataset root. Check structure.")
        return
    print("Using folders:", cat_folder, dog_folder)

    # load balanced dataset
    X_cat, fn_cat = load_images(cat_folder, MAX_PER_CLASS, target_hw)
    X_dog, fn_dog = load_images(dog_folder, MAX_PER_CLASS, target_hw)
    if len(X_cat) == 0 or len(X_dog) == 0:
        print("No images loaded. Check folder contents and MAX_PER_CLASS.")
        return

    X = np.concatenate([X_cat, X_dog], axis=0)
    y = np.concatenate([np.zeros(len(X_cat), dtype=int), np.ones(len(X_dog), dtype=int)], axis=0)
    fnames = fn_cat + fn_dog
    print(f"Loaded {len(X)} samples ({len(X_cat)} cats, {len(X_dog)} dogs).")

    # create ART BlackBoxClassifier using wrapper.predict as predict_fn
    classifier = BlackBoxClassifier(
        predict_fn=wrapper.predict,
        input_shape=target_hw,
        nb_classes=NB_CLASSES
    )

    # evaluate clean accuracy
    probs_clean = classifier.predict(X)
    preds_clean = np.argmax(probs_clean, axis=1)
    clean_acc = (preds_clean == y).mean()
    print(f"Clean accuracy: {clean_acc*100:.2f}% ({(preds_clean==y).sum()}/{len(y)})")

    # Run HopSkipJump (black-box)
    print("Starting HopSkipJump attack (black-box). This may take some time depending on budgets...")
    attack = HopSkipJump(
    classifier=classifier,
    max_iter=HSJ_MAX_ITER,
    max_eval=HSJ_MAX_EVAL,
    init_eval=HSJ_INIT_EVAL,
    init_size=10,
    verbose=True
)


    X_adv = attack.generate(X)

    # Evaluate adversarial accuracy
    probs_adv = classifier.predict(X_adv)
    preds_adv = np.argmax(probs_adv, axis=1)
    adv_acc = (preds_adv == y).mean()
    print(f"Adversarial accuracy: {adv_acc*100:.2f}% ({(preds_adv==y).sum()}/{len(y)})")

    # Save side-by-side
    os.makedirs(SAVE_DIR, exist_ok=True)
    nsave = min(20, len(X_adv))
    for i in range(nsave):
        basename = os.path.splitext(fnames[i])[0] if i < len(fnames) else f"sample_{i}"
        out_path = os.path.join(SAVE_DIR, f"{basename}_idx{i}.png")
        save_side_by_side(X[i], X_adv[i], out_path)

    print("Saved adversarial examples (left original, right adversarial) to", SAVE_DIR)

if __name__ == "__main__":
    main()
