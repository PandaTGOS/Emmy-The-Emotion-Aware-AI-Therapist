import time
import threading
from collections import Counter

VALENCE = {
    "happy": 1.0, "surprise": 0.4, "neutral": 0.0,
    "sad": -1.0, "angry": -0.9, "fear": -0.8, "disgust": -0.9,
    "unknown": 0.0
}

class EmotionAggregator:
    def __init__(self, window_seconds=60, decay_half_life=15):
        self.window = float(window_seconds)
        self.half_life = float(decay_half_life)
        self.history = []   # list of (t, emotion)
        self.lock = threading.Lock()

    def add_emotion(self, emotion, ts=None):
        """Add a detected emotion with optional timestamp"""
        t = ts or time.time()
        with self.lock:
            self.history.append((t, str(emotion)))
            self._prune_locked()

    def _prune_locked(self):
        cutoff = time.time() - self.window
        self.history = [(t, e) for (t, e) in self.history if t >= cutoff]

    def distribution(self):
        """Return counts of emotions in history window"""
        with self.lock:
            self._prune_locked()
            return Counter(e for _, e in self.history)

    def summarize_dict(self):
        """Legacy: return a simple dict summary"""
        dist = self.distribution()
        total = sum(dist.values())
        dominant = "unknown" if total == 0 else dist.most_common(1)[0][0]
        return {
            "dominant": dominant,
            "counts": dict(dist),
            "total": total,
            "window_seconds": self.window
        }

    def summarize_raw_json(self):
        """Legacy: return raw JSON-style dict for LLM context"""
        return self.summarize_dict()

    def summarize_interval(self, delta_t):
        """
        Return summed emotions since last reset, normalized by delta_t (seconds).
        """
        dist = self.distribution()
        total = sum(dist.values())
        norm_counts = {k: v / max(delta_t, 1e-6) for k, v in dist.items()}

        return {
            "counts": dict(dist),
            "normalized": norm_counts,
            "total": total,
            "duration_seconds": delta_t
        }

    def reset(self):
        """Clear stored emotions"""
        with self.lock:
            self.history.clear()
