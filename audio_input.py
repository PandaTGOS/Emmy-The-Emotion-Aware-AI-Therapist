# audio_input.py
import collections
import threading
import time
import webrtcvad
import sounddevice as sd
import numpy as np
import queue
import struct
import wave
import io
import sys

class VoiceListener:
    """
    Continuously listens to the microphone, uses VAD to chunk speech into
    utterances, and puts each as WAV bytes into out_queue.
    """
    def __init__(self, out_queue: queue.Queue, shared_state: dict,
                 sample_rate=16000, frame_ms=30, vad_aggressiveness=2):
        self.out_q = out_queue
        self.shared = shared_state
        self.sr = sample_rate
        self.frame_ms = frame_ms
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.frame_len = int(self.sr * self.frame_ms / 1000)  # samples / frame
        self.stream = None
        self.thread = None

        # Tunables
        self.max_utterance_s = 15.0
        self.pad_ms = 300  # keep some trailing silence
        self.silence_frames_end = self.pad_ms // self.frame_ms


    def _bytes_from_frames(self, frames_list):
        # Convert float32 [-1,1] to 16bit PCM bytes
        audio = np.concatenate(frames_list, axis=0)  # shape: (N, 1)
        audio = np.clip(audio, -1.0, 1.0)
        pcm16 = (audio * 32767).astype(np.int16).tobytes()
        return pcm16

    def _write_wav(self, pcm16_bytes):
        bio = io.BytesIO()
        with wave.open(bio, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sr)
            wf.writeframes(pcm16_bytes)
        return bio.getvalue()

    def _frames(self, indata, frames, time_info, status):
        if status:
            print("Audio status:", status, file=sys.stderr)
        # store raw mono samples (not whole arrays)
        self.buffer.extend(indata[:, 0])  # push individual float samples


    def _run(self):
        self.buffer = collections.deque()

        self.stream = sd.InputStream(
            samplerate=self.sr, channels=1, dtype='float32',
            blocksize=self.frame_len, callback=self._frames
        )
        self.stream.start()

        speech_frames = []
        voiced = False
        silence_run = 0
        utter_start_t = None

        try:
            while not self.shared.get("stop"):
                if len(self.buffer) < self.frame_len:
                    time.sleep(0.001)
                    continue

                # Take exactly frame_len samples (30ms at 16kHz = 480 samples)
                samples = [self.buffer.popleft() for _ in range(self.frame_len)]
                frame = np.array(samples, dtype=np.float32).reshape(-1, 1)

                pcm16 = (np.clip(frame, -1, 1) * 32767).astype(np.int16).tobytes()
                try:
                    is_speech = self.vad.is_speech(pcm16, self.sr)
                except Exception:
                    is_speech = False  # if bad frame

                if is_speech:
                    speech_frames.append(frame)
                    voiced = True
                    silence_run = 0
                    if utter_start_t is None:
                        utter_start_t = time.time()
                else:
                    if voiced:
                        silence_run += 1
                        speech_frames.append(frame)  # trailing silence
                        end_by_sil = silence_run >= self.silence_frames_end
                        end_by_len = (
                            utter_start_t is not None and
                            (time.time() - utter_start_t) >= self.max_utterance_s
                        )
                        if end_by_sil or end_by_len:
                            pcm = self._bytes_from_frames(speech_frames)
                            wav_bytes = self._write_wav(pcm)
                            self.out_q.put(wav_bytes)
                            speech_frames.clear()
                            voiced = False
                            silence_run = 0
                            utter_start_t = None

        finally:
            if self.stream:
                try:
                    self.stream.stop()
                    self.stream.close()
                except Exception:
                    pass


    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
