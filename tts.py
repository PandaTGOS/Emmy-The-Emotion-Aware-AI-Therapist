import platform
import subprocess
import threading
import queue

try:
    import pyttsx3
except Exception:
    pyttsx3 = None


class Speaker:
    def __init__(self, voice: str = None):
        self.q = queue.Queue()
        self.stop_flag = False
        self.is_speaking = False
        self.voice = voice or "Samantha"  # default to natural female on Mac

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

        self.engine = None
        if pyttsx3 and platform.system() != "Darwin":
            try:
                self.engine = pyttsx3.init()
                if voice:
                    self.engine.setProperty("voice", voice)
            except Exception:
                self.engine = None

    def say(self, text: str):
        if not text:
            return
        self.q.put(text)

    def _mac_say(self, text):
        try:
            self.is_speaking = True
            subprocess.run(["say", "-v", self.voice, text], check=False)
        finally:
            self.is_speaking = False

    def _pyttsx3_say(self, text):
        if not self.engine:
            return
        try:
            self.is_speaking = True
            self.engine.say(text)
            self.engine.runAndWait()
        finally:
            self.is_speaking = False

    def _run(self):
        while not self.stop_flag:
            try:
                text = self.q.get(timeout=0.1)
            except queue.Empty:
                continue

            if platform.system() == "Darwin":
                self._mac_say(text)
            else:
                self._pyttsx3_say(text)

    def stop(self):
        self.stop_flag = True
