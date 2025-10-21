# NLP_chatbot — AssemblyAI Integration

This repository is a Streamlit-based voice assistant that was migrated from Picovoice STT/TTS to AssemblyAI's REST APIs for speech-to-text and text-to-speech.

## Environment

You must set the following environment variables (for example, in a `.env` file):

- GEMINI_API_KEY — API key for Google Generative AI (used for responses)
- ASSEMBLYAI_API_KEY — Your AssemblyAI API key (free trial available at https://www.assemblyai.com)

## Install

It's recommended to use a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Notes

- AssemblyAI endpoints used:
  - `POST https://api.assemblyai.com/v2/upload` to upload audio
  - `POST https://api.assemblyai.com/v2/transcript` to create transcripts
  - `POST https://api.assemblyai.com/v2/tts` to generate TTS

- The app uploads recorded audio to AssemblyAI, polls for the transcription, and then uses AssemblyAI TTS to synthesize assistant responses.

- AssemblyAI free trial limits apply — refer to their dashboard for current quotas.

## Quick test (Python)

You can test AssemblyAI upload/transcribe outside the app with:

```python
import requests
headers = {"authorization": "YOUR_KEY"}
# Upload
with open("sample.wav", "rb") as f:
    resp = requests.post("https://api.assemblyai.com/v2/upload", headers=headers, data=f)
    print(resp.json())
```

Replace `YOUR_KEY` with your `ASSEMBLYAI_API_KEY`.
