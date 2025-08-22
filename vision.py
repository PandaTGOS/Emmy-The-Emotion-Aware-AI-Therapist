import cv2
from deepface import DeepFace
from dotenv import load_dotenv

load_dotenv()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def analyze_frame(frame):
    """
    Returns list of face dicts:
      {"box": {"x":..,"y":..,"w":..,"h":..}, "dominant_emotion": "...", "scores": {...}}
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    out = []
    for (x, y, w, h) in detections:
        crop = frame[y:y+h, x:x+w]
        try:
            analysis = DeepFace.analyze(
                crop,
                actions=["emotion"],
                enforce_detection=False
            )
            if isinstance(analysis, list):
                analysis = analysis[0]
            dominant = analysis.get("dominant_emotion") or analysis.get("emotion", {}).keys()
            scores = analysis.get("emotion", {})
        except Exception:
            dominant = "unknown"
            scores = {}

        out.append({
            "box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            "dominant_emotion": dominant,
            "scores": scores
        })

    return out

if __name__ == "__main__":
    # Quick local test: capture one frame and print results
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("webcam not available")
    else:
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("no frame")
        else:
            faces = analyze_frame(frame)
            print("vision test output:", faces)
