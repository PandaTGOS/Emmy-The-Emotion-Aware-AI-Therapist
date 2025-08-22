# Emmy – AI-Powered Emotional Companion  

Emmy is an **AI-driven therapeutic assistant** that combines **computer vision**, **speech processing**, and **natural language conversation** to engage with users in a more human-like and empathetic way.  

- 🎤 **Voice Communication** – Speak naturally, Emmy listens and responds.  
- 👁️ **Emotion Detection** – Real-time facial emotion recognition from webcam.  
- 🧠 **Context-Aware Conversation** – Integrates detected emotions into dialogue.  
- 🗣️ **Text-to-Speech (TTS)** – Emmy speaks responses back for a natural flow.  

Designed as a prototype for **mental health tech** and **empathetic AI assistants**.  

---

## ✨ Features  

- **Real-Time Emotion Tracking**  
  - Uses webcam + deep learning to detect dominant emotions in facial expressions.  
  - Maintains emotion history with decay-based aggregation for context.  

- **Voice Interaction**  
  - Microphone + VAD (Voice Activity Detection) for hands-free conversations.  
  - Automatic **speech-to-text (STT)** transcription.  

- **Conversational AI**  
  - Integrates with an LLM to generate empathetic, context-aware responses.  
  - Emotional state summaries influence replies.  

- **Text-to-Speech (TTS)**  
  - Natural voice synthesis for responses.  
  - Smooth back-and-forth flow without typing.  

- **Visual Feedback**  
  - Live video feed overlays emotion detection and assistant replies.  

---

## 🏗️ Architecture  

```
             🎤 Voice Input       👁️ Video Input
                │                     │
                ▼                     ▼
        ┌───────────────┐     ┌───────────────┐
        │   Voice VAD   │     │   Face + CNN  │
        │   (WebRTC)    │     │   EmotionNet  │
        └───────┬───────┘     └───────┬───────┘
                ▼                     ▼
           Speech-to-Text        EmotionAggregator
                │                     │
                └─────────┬───────────┘
                          ▼
                     Chat Engine
                (LLM + Emotion Context)
                          │
                          ▼
                   Text-to-Speech (TTS)
                          │
                          ▼
                     Spoken Response
```

---

## 🚀 Getting Started  

### Prerequisites  

- **Python 3.9+**  
- Recommended: create a virtual environment  

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows
```

### Installation  

```bash
pip install -r requirements.txt
```

Key dependencies:  
- `opencv-python` – video streaming  
- `mtcnn` – face detection  
- `fer` – emotion recognition  
- `webrtcvad` – voice activity detection  
- `sounddevice` – audio input  
- `numpy`, `scipy` – signal processing  
- `transformers` or `openai` – LLM API for chat  
- `pyttsx3` / cloud TTS – text-to-speech  

### Run  

```bash
python main.py
```

- **Type or Speak**: Emmy listens when you talk.  
- **Webcam Feed**: Displays live video with emotions + replies overlayed.  
- Exit with `quit` or `Ctrl+C`.  

---

## 📂 Project Structure  

```
├── main.py                # Entry point: orchestrates everything
├── audio_input.py         # Microphone + VAD pipeline
├── stt.py                 # Speech-to-text wrapper
├── tts.py                 # Text-to-speech speaker
├── vision.py              # Webcam + face emotion analysis
├── emotion_aggregator.py  # Tracks emotion over time
├── chat.py                # Conversational AI logic
├── utils.py               # Visualization + helpers
├── requirements.txt       
└── README.md
```

---

## 🔮 Future Improvements  

- Multimodal emotion detection (voice tone + facial expressions)  
- More natural conversational memory across sessions  
- Personalization (adapts to user’s baseline emotions)  
- Integration with therapeutic frameworks for structured guidance  
- Web or mobile deployment  

---

## 💡 Why This Project Matters  

Mental health care faces **barriers of access, stigma, and cost**. While AI companions can’t replace therapists, they can:  

- Provide **24/7 empathetic support**  
- Encourage users to express emotions  
- Serve as a **bridge** to professional help  

Emmy is a **demonstration of applied AI** at the intersection of:  
- **Human-Computer Interaction**  
- **Machine Learning for Affect Recognition**  
- **Conversational AI**  

---

## 🧑‍💻 Author  

**Sakhi Saswat Panda**  
- Passionate about **AI for mental health, HCI, and applied ML**  
- Building empathetic AI systems that combine **vision + audio + NLP**  
