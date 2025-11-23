# object_detection_pen_pencil.py
# Updated from user's original script. See original: :contentReference[oaicite:1]{index=1}
import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import time

def detect_with_yolo(model, frame, conf=0.5, desired_classes=None):
    """
    Run YOLO inference on a single frame and return filtered boxes, scores, class names.
    desired_classes: set or list of class-names to show (e.g. {"pen","pencil"})
                    If None -> show all detections.
    """
    results = model.predict(frame, conf=conf, verbose=False)
    r = results[0]

    boxes = []
    if hasattr(r, 'boxes') and r.boxes is not None:
        for box in r.boxes:
            xyxy = box.xyxy[0].cpu().numpy()   # x1,y1,x2,y2
            conf_score = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = model.names.get(cls_id, str(cls_id))

            if desired_classes is None or cls_name.lower() in desired_classes:
                boxes.append({
                    "xyxy": xyxy.astype(int).tolist(),
                    "conf": conf_score,
                    "class": cls_name
                })
    return boxes

def detect_long_objects_opencv(frame, min_area=1500, min_aspect=3.0):
    """
    Heuristic fallback to detect elongated objects (pens/pencils) by contour shape.
    Returns list of boxes (x1,y1,x2,y2) for elongated contours.
    Not a replacement for training a detector, but useful for quick testing.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7,7), 0)
    # Use adaptive threshold / Canny to catch edges
    edged = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # ensure elongated: aspect = longer/shorter
        aspect = max(w/h, h/w) if h>0 and w>0 else 0
        if aspect >= min_aspect:
            # optional: area ratio check (contour area vs box area)
            box_area = w*h
            if box_area == 0:
                continue
            fill_ratio = area / float(box_area)
            # pens are often thin: fill_ratio not too large (but this is heuristic)
            if fill_ratio < 0.9:
                boxes.append([x, y, x+w, y+h])
    return boxes

def draw_boxes(frame, boxes, label_prefix="YOLO"):
    for b in boxes:
        if isinstance(b, dict):
            x1,y1,x2,y2 = b["xyxy"]
            label = f"{label_prefix}:{b['class']} {b['conf']:.2f}"
        else:
            x1,y1,x2,y2 = b
            label = f"{label_prefix}:elongated"
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

def main(source=0, model_name="yolov8n.pt", conf=0.5, use_yolo=True):
    """
    - source: webcam index (0) or path to video/image
    - model_name: path to a trained model (.pt) OR a pretrained weight name like 'yolov8n.pt'
    - conf: YOLO confidence threshold
    - use_yolo: try YOLO first. If YOLO does not contain desired classes, fallback to OpenCV heuristic.
    """
    # Define desired classes you want to detect (lowercase)
    # If you have trained a custom model with these labels, put them here exactly.
    desired_classes = {"pen", "pencil", "marker", "crayon", "highlighter"}

    # Load model (pretrained or custom). If you trained 'best.pt' place its path here.
    print(f"[INFO] Loading model: {model_name}")
    model = YOLO(model_name)

    # Check if model has the desired classes:
    model_class_names = {v.lower() for k,v in model.names.items()}
    has_desired = len(desired_classes.intersection(model_class_names)) > 0
    if has_desired:
        print("[INFO] Model appears to contain desired classes: using YOLO detections for pens/pencils.")
    else:
        print("[WARN] Model does NOT appear to contain 'pen'/'pencil' labels.")
        print("       YOLO will still run (may detect close objects like 'scissors' etc.),")
        print("       but fallback OpenCV heuristic will also be used for elongated shapes.")

    # If source is a single image file, run once
    if isinstance(source, str) and source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        frame = cv2.imread(source)
        if frame is None:
            raise RuntimeError(f"Could not read image: {source}")

        yolo_boxes = detect_with_yolo(model, frame, conf=conf, desired_classes=desired_classes if has_desired else None)
        draw_boxes(frame, yolo_boxes, label_prefix="YOLO")

        if not has_desired:
            fallback = detect_long_objects_opencv(frame)
            draw_boxes(frame, fallback, label_prefix="FALLBACK")

        cv2.imshow("Detections", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # video/webcam stream
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    # Optional: set webcam resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    fps_time = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        yolo_boxes = []
        fallback_boxes = []

        if use_yolo:
            try:
                yolo_boxes = detect_with_yolo(model, frame, conf=conf, desired_classes=desired_classes if has_desired else None)
            except Exception as e:
                # In case of unexpected model error, fallback to OpenCV
                print("[ERROR] YOLO inference failed:", e)

        if (not has_desired) or (len(yolo_boxes) == 0):
            # fallback heuristic for elongated objects
            fallback_boxes = detect_long_objects_opencv(frame)

        # Draw detections
        draw_boxes(frame, yolo_boxes, label_prefix="YOLO")
        draw_boxes(frame, fallback_boxes, label_prefix="FALLBACK")

        # FPS display
        fps = 1.0 / (time.time() - fps_time + 1e-6)
        fps_time = time.time()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        cv2.imshow("Pen/Pencil Detector (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect pens/pencils using YOLOv8 + OpenCV fallback.")
    parser.add_argument("--source", type=str, default="0", help="0 for webcam or path to video/image")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Ultralytics model or path to custom best.pt")
    parser.add_argument("--conf", type=float, default=0.35, help="confidence threshold for YOLO")
    parser.add_argument("--no-yolo", action="store_true", help="disable YOLO and use only OpenCV heuristic")
    args = parser.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    main(source=src, model_name=args.model, conf=args.conf, use_yolo=(not args.no_yolo))
