# detection_helpers.py
import os, time, csv, hashlib
from collections import defaultdict, deque
from datetime import datetime
from io import BytesIO
from PIL import Image
import imagehash

# --- configuration (tune thresholds if needed) ---
RATE_WINDOW = 60          # seconds
RATE_THRESHOLD = 60       # max requests per minute per client
DUP_WINDOW = 300          # seconds
DUP_THRESHOLD = 10        # same or near-duplicate images in window
LOW_CONF_WIN = 300        # seconds
LOW_CONF_THRESHOLD = 15   # many low-confidence queries in window
LOW_CONF_PROB = 0.6       # below this = low confidence
NEAR_DUP_SIM = 0.98       # similarity threshold for near duplicates

query_log_path = "query_log.csv"
alerts_log = "query_alerts.log"

# in-memory state
client_times = defaultdict(deque)
client_hashes = defaultdict(deque)
client_lowconf = defaultdict(deque)
client_phashes = defaultdict(deque)

def _now(): return time.time()

def compute_sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()

def compute_phash(image_bytes):
    img = Image.open(BytesIO(image_bytes)).convert("RGB").resize((150,150))
    return imagehash.phash(img)

def log_query(client_id, filename, img_hash, pred, conf):
    row = {
        "ts": datetime.utcnow().isoformat(),
        "client": client_id,
        "file": filename,
        "image_hash": img_hash,
        "pred": pred,
        "conf": conf,
    }
    write_header = not os.path.exists(query_log_path)
    with open(query_log_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)

def log_alert(msg):
    msg_full = f"{datetime.utcnow().isoformat()} ALERT: {msg}"
    print(msg_full)
    with open(alerts_log, "a") as f:
        f.write(msg_full + "\n")

def check_rules(client_id, img_hash, phash, conf):
    now = _now()
    # --- rate limiting ---
    t = client_times[client_id]
    t.append(now)
    while t and t[0] < now - RATE_WINDOW:
        t.popleft()
    if len(t) > RATE_THRESHOLD:
        log_alert(f"High query rate ({len(t)} req/min) from {client_id}")

    # --- duplicate hashes ---
    hlist = client_hashes[client_id]
    hlist.append((now, img_hash))
    while hlist and hlist[0][0] < now - DUP_WINDOW:
        hlist.popleft()
    dup_count = sum(1 for _, h in hlist if h == img_hash)
    if dup_count >= DUP_THRESHOLD:
        log_alert(f"Duplicate image probing by {client_id}")

    # --- near-duplicate hashes ---
    phlist = client_phashes[client_id]
    phlist.append((now, phash))
    while phlist and phlist[0][0] < now - DUP_WINDOW:
        phlist.popleft()
    similar = 0
    for _, p in phlist:
        sim = 1.0 - (phash - p) / phash.hash.size
        if sim >= NEAR_DUP_SIM:
            similar += 1
    if similar >= DUP_THRESHOLD:
        log_alert(f"Near-duplicate probing ({similar}) by {client_id}")

    # --- low confidence ---
    if conf < LOW_CONF_PROB:
        lq = client_lowconf[client_id]
        lq.append(now)
        while lq and lq[0] < now - LOW_CONF_WIN:
            lq.popleft()
        if len(lq) >= LOW_CONF_THRESHOLD:
            log_alert(f"Many low-confidence queries by {client_id}")
