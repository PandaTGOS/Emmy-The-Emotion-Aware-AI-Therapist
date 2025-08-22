import cv2

def draw_faces_and_reply(frame, faces, reply_text):
    for face in faces:
        box = face.get("box", {})
        if not box:
            continue
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label = face.get("dominant_emotion", "unknown")
        cv2.putText(frame, label, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    if reply_text:
        # clip long replies
        r = reply_text if len(reply_text) <= 120 else reply_text[:117] + "..."
        cv2.putText(frame, "Emmy: " + r, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
