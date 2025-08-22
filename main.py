# main.py
import threading
import time
import cv2
import json
import signal
import sys
import queue

from vision import analyze_frame
from emotion_aggregator import EmotionAggregator
from chat import chat
from utils import draw_faces_and_reply

from audio_input import VoiceListener
from stt import init_stt, transcribe_wav_bytes
from tts import Speaker


def vision_loop(aggregator, shared_state):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("webcam open failed")
        return

    target_interval = 0.5  # ~2fps
    while not shared_state.get("stop"):
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        faces = analyze_frame(frame)
        for f in faces:
            agg_em = f.get("dominant_emotion") or "unknown"
            aggregator.add_emotion(agg_em)

        with shared_state["lock"]:
            shared_state["latest_frame"] = frame.copy()
            shared_state["latest_faces"] = faces
            shared_state["latest_reply"] = shared_state.get("latest_reply", "")

        elapsed = time.time() - t0
        to_sleep = target_interval - elapsed
        if to_sleep > 0:
            time.sleep(to_sleep)

    cap.release()


def handle_sigint(shared, speaker):
    print("\n🛑 Caught Ctrl+C, shutting down gracefully...")
    shared["stop"] = True
    try:
        if speaker:
            speaker.stop()
    except Exception:
        pass
    cv2.destroyAllWindows()
    sys.exit(0)


def main():
    init_stt()  # load STT model once

    agg = EmotionAggregator(window_seconds=60, decay_half_life=12)
    shared = {
        "latest_frame": None,
        "latest_faces": [],
        "latest_reply": "",
        "stop": False,
        "lock": threading.Lock()
    }

    # 1) Start Video+Emotion thread
    vt = threading.Thread(target=vision_loop, args=(agg, shared), daemon=True)
    vt.start()

    # 2) Start Mic+VAD listener -> wav_queue
    wav_q = queue.Queue()
    voice = VoiceListener(out_queue=wav_q, shared_state=shared)
    voice.start()

    # 3) TTS speaker
    speaker = Speaker(voice="Allison")

    # Ctrl+C handler
    signal.signal(signal.SIGINT, lambda s, f: handle_sigint(shared, speaker))

    print("🎤 Voice mode active. Speak to Emmy. Type 'quit' to exit (or Ctrl+C).")
    print("Tip: You can still type if you prefer.")

    last_chat_time = time.time()

    try:
        while not shared.get("stop"):
            # Either: A) user typed, or B) we got a voice utterance
            typed_text = None
            if sys.stdin in select_inputs():
                try:
                    typed_text = input("\nYou: ").strip()
                except EOFError:
                    typed_text = None

            user_text = None
            if typed_text:
                if typed_text.lower() in ("quit", "exit"):
                    break
                user_text = typed_text
            else:
                # Only listen if not speaking
                if not speaker.is_speaking:
                    try:
                        wav_bytes = wav_q.get(timeout=0.2)
                    except queue.Empty:
                        wav_bytes = None

                    if wav_bytes:
                        # STT
                        user_text = transcribe_wav_bytes(wav_bytes)
                        if user_text:
                            print(f"\nYou (voice): {user_text}")


            # If we have something to send to LLM:
            if user_text:
                now = time.time()
                delta_t = now - last_chat_time
                last_chat_time = now

                emotion_summary = agg.summarize_interval(delta_t)
                print("🟢 Sending with emotions:", json.dumps(emotion_summary, indent=2))

                reply = chat(user_text, emotion_summary)
                if not reply:
                    reply = "(error: no response from LLM)"
                print("Emmy:", reply)

                # speak it
                speaker.say(reply)

                # reset emotions for next turn
                agg.reset()

                # update overlay
                with shared["lock"]:
                    shared["latest_reply"] = reply

            # refresh display if we have a frame
            with shared["lock"]:
                frame = shared.get("latest_frame")
                faces = list(shared.get("latest_faces", []))
                rply = shared.get("latest_reply", "")
            if frame is not None:
                draw_faces_and_reply(frame, faces, rply)
                cv2.imshow("Emmy - Live", frame)
                cv2.waitKey(1)

    finally:
        shared["stop"] = True
        speaker.stop()
        cv2.destroyAllWindows()


def select_inputs():
    """
    Non-blocking check if there's something to read from stdin.
    Works on Unix/macOS. On Windows, you can skip typed input or use msvcrt.
    """
    try:
        import select
        r, _, _ = select.select([sys.stdin], [], [], 0)
        return r
    except Exception:
        return []


if __name__ == "__main__":
    main()
