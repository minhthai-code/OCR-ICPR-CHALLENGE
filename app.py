"""
PlateVision Backend — Flask API
Replicates the exact detection pipeline from your script:
  vehicle_model → crop vehicle → lp_model → best LP → crop LP → OCR model
"""
import os
import sys
import time
import base64
import traceback
import threading

import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO

# ── CONFIG ─────────────────────────────────────────────────────────────────────
VEHICLE_MODEL_PATH = r"C:\Users\Admin\Downloads\vehicle_best.pt"
LP_MODEL_PATH      = r"C:\Users\Admin\Downloads\lp_best.pt"
OCR_MODEL_PATH     = r"D:\AIOT PROJET\OCR-MultiFrame-ICPR\pretrain_results\pretrain_stage1_best_81_38.pth"
OCR_PROJECT_ROOT   = r"D:\AIOT PROJET\OCR-MultiFrame-ICPR"

CONF_THRESHOLD = 0.25
LP_PAD_W = 0.10
LP_PAD_H = 0.15

# ── Add project root to sys.path ──────────────────────────────────────────────
sys.path.insert(0, OCR_PROJECT_ROOT)

# ── FLASK ───────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static")
CORS(app)

# ── GLOBALS ────────────────────────────────────────────────────────────────────
vehicle_model = None
lp_model      = None
ocr_model     = None
ocr_config    = None
device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_lock    = threading.Lock()

# ── HELPER: build_confusion_groups (copied from train_phase.py) ──────────────
def build_confusion_groups(config) -> list:
    raw_groups = [
        ("O", "D", "Q"), ("M", "N", "H"), ("6", "8", "4", "9"),
        ("2", "3"), ("V", "Y"), ("A", "B"), ("E", "C"), ("W", "V"), ("1", "7"),
    ]
    confusion_groups = []
    for group in raw_groups:
        indices = [config.CHAR2IDX[ch] for ch in group if ch in config.CHAR2IDX]
        if len(indices) >= 2:
            confusion_groups.append(indices)
    return confusion_groups

# ── DRAW HELPERS ──────────────────────────────────────────────────────────────
def draw_modern_bbox(img, pt1, pt2, color, is_vehicle=True):
    x1, y1 = pt1
    x2, y2 = pt2
    img_h, img_w = img.shape[:2]
    base_thick = max(1, int(img_w / 900))

    if is_vehicle:
        thickness = int(base_thick * 1.2)
        corner_thickness = int(base_thick * 3)
        corner_len = int(max(x2 - x1, y2 - y1) * 0.06)
        glow_thickness = int(thickness * 3)
    else:
        thickness = int(base_thick * 0.6)
        corner_thickness = int(base_thick * 2)
        corner_len = int(max(x2 - x1, y2 - y1) * 0.10)
        glow_thickness = int(thickness * 4)

    corner_len = max(6, corner_len)

    # Glow effect (semi‑transparent wider line behind)
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, glow_thickness)
    cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)

    # Main rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    # Corner brackets
    cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, corner_thickness)
    cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, corner_thickness)
    cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, corner_thickness)
    cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, corner_thickness)
    cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, corner_thickness)
    cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, corner_thickness)
    cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, corner_thickness)
    cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, corner_thickness)

# ── MODEL LOADING ──────────────────────────────────────────────────────────────
def load_models():
    global vehicle_model, lp_model, ocr_model, ocr_config

    print("🔄 Loading YOLO models...")
    vehicle_model = YOLO(VEHICLE_MODEL_PATH)
    lp_model = YOLO(LP_MODEL_PATH)
    print("✅ YOLO models loaded!")

    # Load config (use the same as training – final_config or ablation_config)
    from configs.final_config import Config
    ocr_config = Config()

    print("🔄 Loading OCR checkpoint...")
    checkpoint = torch.load(OCR_MODEL_PATH, map_location=device)

    # Extract student state dict
    if "student" in checkpoint:
        state_dict = checkpoint["student"]
        print("   Using 'student' weights.")
    elif "teacher" in checkpoint:
        state_dict = checkpoint["teacher"]
        print("   Using 'teacher' weights.")
    else:
        state_dict = {k: v for k, v in checkpoint.items()
                      if not k.startswith("epoch") and not k.startswith("optimizer")}
        print("   Using full checkpoint.")

    # Strip any "student." or "teacher." prefix
    if any(k.startswith("student.") for k in state_dict.keys()):
        state_dict = {k.replace("student.", "", 1): v for k, v in state_dict.items()}
        print("   Stripped 'student.' prefix.")
    elif any(k.startswith("teacher.") for k in state_dict.keys()):
        state_dict = {k.replace("teacher.", "", 1): v for k, v in state_dict.items()}
        print("   Stripped 'teacher.' prefix.")

    # ---- REMAP FUSION KEYS: temporal_fusion -> fusion ----
    remapped = {}
    for k, v in state_dict.items():
        if k.startswith("temporal_fusion."):
            new_k = "fusion." + k[len("temporal_fusion."):]
            remapped[new_k] = v
        else:
            remapped[k] = v
    state_dict = remapped
    print("   Remapped 'temporal_fusion' keys to 'fusion'.")

    # ---- DETECT BACKBONE ----
    all_keys = list(state_dict.keys())
    has_proposed = any("backbone.stage1" in k or "backbone.transformer" in k for k in all_keys)
    has_resnet = any("backbone.conv1" in k or "backbone.layer1" in k for k in all_keys)
    has_svtr = any("backbone.patch_embed" in k for k in all_keys)

    if has_proposed:
        backbone_name = "proposed"
    elif has_svtr:
        backbone_name = "svtrv2"
    elif has_resnet:
        backbone_name = "resnet34"
    else:
        backbone_name = "proposed"  # fallback
    print(f"   Deduced backbone: {backbone_name}")

    # ---- DEDUCE PARAMETERS ----
    if "head.slot_embed.weight" in state_dict:
        slot_embed_shape = state_dict["head.slot_embed.weight"].shape
        num_slots = slot_embed_shape[0]
        slot_dim = slot_embed_shape[1]
    else:
        num_slots = ocr_config.NUM_SLOTS
        slot_dim = ocr_config.SLOT_DIM
    print(f"   Deduced num_slots={num_slots}, slot_dim={slot_dim}")

    use_refinement = any("risk_gate" in k or "confusion_refiner" in k for k in all_keys)
    use_stn = any("stn" in k for k in all_keys)
    is_multiframe = any("fusion" in k for k in all_keys)  # after remap, we have fusion keys
    use_motion_alignment = any("motion_alignment" in k for k in all_keys)

    fusion_name = "transformer"  # the checkpoint used transformer fusion
    pretrained_backbone = False if backbone_name == "proposed" else True

    # ---- IMPORTANT: Set confusion_groups = None to match the checkpoint's [1, 37] matrix ----
    confusion_groups = None   # this will create a single group matrix

    print(f"🔍 Final flags: use_refinement={use_refinement}, "
          f"is_multiframe={is_multiframe}, backbone={backbone_name}, "
          f"fusion={fusion_name}, use_stn={use_stn}, num_slots={num_slots}")

    # ---- Instantiate model with deduced parameters ----
    from src.models.restran import ResTranOCR

    ocr_model = ResTranOCR(
        num_classes=ocr_config.NUM_CLASSES,
        num_slots=num_slots,
        cnn_channels=slot_dim,
        transformer_heads=ocr_config.TRANSFORMER_HEADS,
        transformer_layers=ocr_config.TRANSFORMER_LAYERS,
        transformer_ff_dim=ocr_config.TRANSFORMER_FF_DIM,
        dropout=ocr_config.TRANSFORMER_DROPOUT,
        use_stn=use_stn,
        pretrained_backbone=pretrained_backbone,
        use_refinement=use_refinement,
        confusion_groups=confusion_groups,   # None → single group
        backbone_name=backbone_name,
        openocr_root=ocr_config.OPENOCR_ROOT,
        svtrv2_weights_url=ocr_config.SVTRV2_WEIGHTS_URL,
        freeze_backbone=False,
        use_motion_alignment=use_motion_alignment,
        is_multiframe=is_multiframe,
        fusion_name=fusion_name,
        backbone_weights_path=None,
    ).to(device)

    # ---- Load weights ----
    try:
        ocr_model.load_state_dict(state_dict, strict=True)
        print("✅ OCR model loaded with strict matching!")
    except RuntimeError as e:
        print(f"⚠️  Strict loading failed: {e}")
        print("   Trying with strict=False...")
        ocr_model.load_state_dict(state_dict, strict=False)
        print("✅ OCR model loaded with non-strict matching (some layers may be uninitialized).")

    ocr_model.eval()
    print("✅ OCR model ready!")

# ── OCR INFERENCE ──────────────────────────────────────────────────────────────
def run_ocr(lp_crop_bgr):
    """Run OCR on a BGR license plate crop. Returns (text, confidence)."""
    if ocr_model is None:
        return "NO_OCR", 0.0

    try:
        h = ocr_config.IMG_HEIGHT
        w = ocr_config.IMG_WIDTH
        idx2char = ocr_config.IDX2CHAR

        # Preprocess single frame: BGR → RGB → resize → normalize
        rgb = cv2.cvtColor(lp_crop_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0

        # Use the same normalization as training (0.5 mean/std is common)
        mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        std  = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        tensor = tensor.unsqueeze(0).to(device)   # [1, C, H, W]

        # Multi-frame: repeat the single frame 5 times → [1, 5, C, H, W]
        is_multi = getattr(ocr_model, "is_multiframe", False)
        if is_multi:
            frames = torch.stack([tensor] * 5, dim=1)   # [1, 5, C, H, W]
        else:
            frames = tensor

        # Forward pass
        with torch.no_grad():
            output = ocr_model(frames)   # could be tensor, tuple, or dict

        # Extract logits
        if isinstance(output, dict):
            # If return_aux was used, logits may be under 'logits'
            logits = output.get('logits', output.get('base_logits'))
            if logits is None:
                # fallback: take any tensor in dict
                for v in output.values():
                    if isinstance(v, torch.Tensor) and v.dim() == 3:
                        logits = v
                        break
                else:
                    raise ValueError("Could not find logits in dict output")
        elif isinstance(output, (list, tuple)):
            # Usually the first element is logits
            logits = output[0]
        else:
            logits = output

        # logits shape: [B, num_slots, num_classes]
        if logits.dim() == 3:
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)      # [B, num_slots]
            conf_per_slot = probs.max(dim=-1)[0]  # [B, num_slots]

            # Decode only the first sample (B=1)
            pred = preds[0]
            confs = conf_per_slot[0]
            text = ""
            scores = []
            for slot_idx, p in enumerate(pred):
                if p != 0:   # 0 is blank/pad
                    ch = idx2char.get(int(p), "?")
                    text += ch
                    scores.append(float(confs[slot_idx].item()))
            avg_conf = float(np.mean(scores)) * 100 if scores else 0.0

            # 🔧 FIX: Enforce Brazilian plate standard – max 7 characters
            if len(text) > 7:
                text = text[:7]
                # Optionally recalc avg_conf for truncated chars? Not necessary,
                # but we can keep the original avg_conf (it's fine).

            return text.upper() if text else "?", avg_conf
        else:
            raise ValueError(f"Unexpected logits shape: {logits.shape}")

    except Exception as e:
        print(f"OCR inference error: {e}")
        import traceback
        traceback.print_exc()
        return "ERR", 0.0

# ── FRAME DETECTION ──────────────────────────────────────────────────────────────
def detect_frame(frame_bgr, conf_threshold=CONF_THRESHOLD):
    height, width = frame_bgr.shape[:2]
    annotated = frame_bgr.copy()
    detections = []

    with model_lock:
        vehicle_results = vehicle_model(frame_bgr, verbose=False, conf=conf_threshold)[0]

    for vehicle_box in vehicle_results.boxes:
        vx1, vy1, vx2, vy2 = map(int, vehicle_box.xyxy[0].tolist())
        v_conf = float(vehicle_box.conf[0])

        color_v = (0, 150, 255)   # bright blue for vehicles
        draw_modern_bbox(annotated, (vx1, vy1), (vx2, vy2), color_v, is_vehicle=True)

        vx1c = max(0, vx1); vy1c = max(0, vy1)
        vx2c = min(width, vx2); vy2c = min(height, vy2)
        vehicle_crop = frame_bgr[vy1c:vy2c, vx1c:vx2c]
        if vehicle_crop.size == 0:
            continue

        with model_lock:
            lp_results = lp_model(vehicle_crop, verbose=False, conf=conf_threshold)[0]

        best_lp_box = None
        max_conf = -1.0
        for lp_box in lp_results.boxes:
            c = lp_box.conf[0].item()
            if c > max_conf:
                max_conf = c
                best_lp_box = lp_box

        if best_lp_box is None:
            detections.append({
                "vehicle": {"x1": vx1, "y1": vy1, "x2": vx2, "y2": vy2, "conf": round(v_conf, 3)},
                "plate": None,
            })
            continue

        lx1, ly1, lx2, ly2 = map(int, best_lp_box.xyxy[0].tolist())
        lp_w = lx2 - lx1
        lp_h = ly2 - ly1
        pad_w = int(lp_w * LP_PAD_W)
        pad_h = int(lp_h * LP_PAD_H)

        global_lx1 = max(0,     vx1c + lx1 - pad_w)
        global_ly1 = max(0,     vy1c + ly1 - pad_h)
        global_lx2 = min(width,  vx1c + lx2 + pad_w)
        global_ly2 = min(height, vy1c + ly2 + pad_h)

        color_lp = (0, 255, 200)  # cyan‑green for plates
        draw_modern_bbox(annotated, (global_lx1, global_ly1), (global_lx2, global_ly2), color_lp, is_vehicle=False)

        lp_crop = frame_bgr[global_ly1:global_ly2, global_lx1:global_lx2]
        ocr_text, ocr_conf = run_ocr(lp_crop) if lp_crop.size > 0 else ("", 0.0)

        label = f"{ocr_text}  {ocr_conf:.0f}%"
        font_scale = max(0.4, width / 2000)
        font_thick = max(1, int(width / 900))
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, font_scale, font_thick)
        tx = global_lx1
        ty = max(global_ly1 - 6, th + 4)
        cv2.rectangle(annotated, (tx, ty - th - bl - 2), (tx + tw + 6, ty + 2), (0, 220, 50), -1)
        cv2.putText(annotated, label, (tx + 3, ty - bl), cv2.FONT_HERSHEY_DUPLEX,
                    font_scale, (0, 0, 0), font_thick, cv2.LINE_AA)

        lp_b64 = ""
        if lp_crop.size > 0:
            _, buf = cv2.imencode(".jpg", lp_crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            lp_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

        detections.append({
            "vehicle": {"x1": vx1, "y1": vy1, "x2": vx2, "y2": vy2, "conf": round(v_conf, 3)},
            "plate": {
                "x1": global_lx1, "y1": global_ly1,
                "x2": global_lx2, "y2": global_ly2,
                "conf": round(max_conf, 3),
                "ocr_text": ocr_text,
                "ocr_conf": round(ocr_conf, 1),
                "crop_b64": lp_b64,
            },
        })

    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
    annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode()
    return annotated_b64, detections

# ── ROUTES ────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/api/status")
def status():
    return jsonify({
        "vehicle_model": vehicle_model is not None,
        "lp_model": lp_model is not None,
        "ocr_model": ocr_model is not None,
        "device": str(device),
        "vehicle_path": VEHICLE_MODEL_PATH,
        "lp_path": LP_MODEL_PATH,
        "ocr_path": OCR_MODEL_PATH,
    })

@app.route("/api/detect", methods=["POST"])
def detect():
    try:
        data = request.get_json(force=True)
        if not data or "frame" not in data:
            return jsonify({"error": "Missing 'frame' field"}), 400

        conf = float(data.get("conf", CONF_THRESHOLD))
        b64 = data["frame"]
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        img_bytes = base64.b64decode(b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Could not decode frame"}), 400

        t0 = time.time()
        annotated_b64, detections = detect_frame(frame, conf_threshold=conf)
        ms = round((time.time() - t0) * 1000, 1)
        return jsonify({"annotated": annotated_b64, "detections": detections, "ms": ms})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/detect_image", methods=["POST"])
def detect_image():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    nparr = np.frombuffer(f.read(), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Cannot decode image"}), 400
    t0 = time.time()
    annotated_b64, detections = detect_frame(frame)
    ms = round((time.time() - t0) * 1000, 1)
    return jsonify({"annotated": annotated_b64, "detections": detections, "ms": ms})

# ── STARTUP ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_models()
    print("\n🚀 PlateVision server running at http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)