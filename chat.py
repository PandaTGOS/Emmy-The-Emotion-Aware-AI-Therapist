import os
import json
import requests
import time

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")
TIMEOUT = 30
MAX_HISTORY = 10  # keep last 10 user+assistant pairs

SYSTEM_PROMPT = (
    "You are Emmy, an empathetic AI therapist. Provide a safe, supportive, non-judgmental environment. "
    "Use reflective listening, validate feelings, ask open-ended questions, and help the user explore feelings. "
    "Consider the conversation history, the provided emotion data between previous and current response as the , and the conversation sentiment when planning the next step. "
    "If there are signs of imminent danger or self-harm, respond with a crisis flow that urges the user to seek immediate help and includes local emergency resources. "
    "Do not provide medical diagnoses. Keep responses compassionate, concise, and therapy-oriented."
    "**KEEP EVERY RESPONSE SHORT AND CONCISE SOMETHING THAT CAN BE SAID QUICKLY**"
)

_history = [{"role": "system", "content": SYSTEM_PROMPT}]

POSITIVE_WORDS = {"good","well","happy","better","great","relieved","hope"}
NEGATIVE_WORDS = {"sad","upset","angry","depressed","stressed","overwhelmed","anxious","bad","lonely"}

def _simple_sentiment_from_history(history):
    """Simple heuristic: ratio of positive-root to negative-root tokens across recent user messages."""
    pos = 0
    neg = 0
    for m in history:
        if m.get("role") != "user":
            continue
        txt = m.get("content","").lower()
        for w in POSITIVE_WORDS:
            if w in txt:
                pos += 1
        for w in NEGATIVE_WORDS:
            if w in txt:
                neg += 1
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total  # -1..1

def _llm_chat(messages, api_key=None, model=None, timeout=TIMEOUT):
    api_key = api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")

    payload = {
        "model": model or DEFAULT_MODEL,
        "messages": messages,
        "temperature": 0.5,
        "max_tokens": 512
    }

    # debug print of what is sent (without disclosing key)
    print("\n--- LLM payload ---")
    print(json.dumps({k:v for k,v in payload.items() if k!="messages" }, indent=2))
    print("messages:")
    print(json.dumps(payload["messages"], indent=2))
    print("--- end payload ---\n")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        r = requests.post(GROQ_URL, json=payload, headers=headers, timeout=timeout)
        r.raise_for_status()
        d = r.json()
        if isinstance(d, dict) and "choices" in d:
            return d["choices"][0]["message"]["content"].strip()
        if isinstance(d, dict) and "response" in d:
            return str(d["response"]).strip()
        return str(d)
    except requests.RequestException as e:
        print("LLM request error:", e)
        return None


def chat(user_text, emotion_summary=None, model=None, api_key=None):
    global _history
    if not _history:
        _history = [{"role":"system","content":SYSTEM_PROMPT}]

    emotion_ctx_text = ""
    if emotion_summary:
        emotion_ctx_text = f"[EmotionContext] {json.dumps(emotion_summary)}"

    sentiment_score = _simple_sentiment_from_history(_history)
    sentiment_text = f"[ConversationSentiment] {round(sentiment_score,3)}"

    # Build fresh context
    messages = [{"role":"system","content":SYSTEM_PROMPT}]
    if emotion_ctx_text:
        messages.append({"role":"system","content":emotion_ctx_text})
    messages.append({"role":"system","content":sentiment_text})

    # add only last N exchanges
    user_assistant_only = [m for m in _history if m["role"] in ("user","assistant")]
    if len(user_assistant_only) > MAX_HISTORY*2:
        user_assistant_only = user_assistant_only[-MAX_HISTORY*2:]

    messages.extend(user_assistant_only)
    messages.append({"role":"user","content": user_text})

    reply = _llm_chat(messages, api_key=api_key, model=model)
    if reply is None:
        reply = "(error: no response from LLM)"

    _history.append({"role":"user","content": user_text})
    _history.append({"role":"assistant","content": reply})

    # Trim history after adding
    user_assistant_only = [m for m in _history if m["role"] in ("user","assistant")]
    if len(user_assistant_only) > MAX_HISTORY*2:
        # keep system + last exchanges
        _history = [{"role":"system","content":SYSTEM_PROMPT}] + user_assistant_only[-MAX_HISTORY*2:]

    return reply



def reset_history():
    global _history
    _history = [{"role":"system","content":SYSTEM_PROMPT}]

if __name__ == "__main__":
    print("CLI chat test. Type 'quit' to exit.")
    while True:
        txt = input("You: ").strip()
        if txt.lower() in ("quit","exit"):
            break
        em = "neutral | quick test"
        r = chat(txt, emotion_summary=em)
        print("Emmy:", r)
