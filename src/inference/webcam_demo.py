import argparse
import time
import cv2
import tensorflow as tf
from src.data.segment_hands import HandCropper
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Local OpenCV ASL webcam demo")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args(); config = load_config(args.config)
    model = tf.keras.models.load_model(config["paths"]["model"])
    mp_cfg = config["mediapipe"]
    cropper = HandCropper(mp_cfg["model_path"], mp_cfg["padding_ratio"], mp_cfg["min_hand_detection_confidence"], mp_cfg["min_hand_presence_confidence"], mp_cfg["min_tracking_confidence"])
    capture, previous = cv2.VideoCapture(args.camera), time.perf_counter()
    try:
        while capture.isOpened():
            ok, frame = capture.read()
            if not ok: break
            crop, status = cropper.crop(frame)
            message = "No hand detected"
            if status == "ok":
                image = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (config["data"]["image_size"],) * 2).astype("float32")[None, ...]
                scores = model.predict(image, verbose=0)[0]; index = scores.argmax()
                message = f"{config['classes'][index]}  {scores[index]:.1%}"
            now = time.perf_counter(); fps = 1 / max(now - previous, 1e-6); previous = now
            cv2.putText(frame, message, (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (18, 76), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
            cv2.imshow("ASL Static Recognition", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")): break
    finally:
        capture.release(); cropper.close(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
