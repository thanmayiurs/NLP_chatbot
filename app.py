from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai
import os
import tempfile
import io
import numpy as np
from scipy.io import wavfile
import subprocess
import re
import base64
import requests  # Required for AssemblyAI STT
import time  # Required for API polling
import uuid
import platform
import pathlib
import streamlit.components.v1 as components

load_dotenv()

# ---------- CONFIG ----------
st.set_page_config(page_title="Gemini Voice Assistant", page_icon="🎤", layout="wide")

# Initialize APIs with error handling
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-2.5-flash', 
                                system_instruction="Keep responses concise and professional.")
    
    genai_api_key = os.environ.get("GEMINI_API_KEY")
    # AssemblyAI key for STT/TTS
    ASSEMBLYAI_API_KEY = os.environ.get("ASSEMBLYAI_API_KEY")
    if not ASSEMBLYAI_API_KEY:
        raise EnvironmentError("ASSEMBLYAI_API_KEY not set in environment")
    ASSEMBLYAI_HEADERS = {"authorization": ASSEMBLYAI_API_KEY}
except Exception as e:
    st.error(f"Failed to initialize APIs: {str(e)}")
    st.stop()

# ---------- STATE ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_audio" not in st.session_state:
    st.session_state.last_audio = None
if "audio_key" not in st.session_state:
    st.session_state.audio_key = 0
if "processing" not in st.session_state:
    st.session_state.processing = False
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

# ---------- UTILS ----------
@st.cache_data(ttl=3600)  # Cache text cleaning for 1 hour
def clean_text(text):
    """Clean text of markdown and other formatting"""
    chars_to_remove = ['**', '*', '__', '_', '`', '~', '#', '[', ']']
    for char in chars_to_remove:
        text = text.replace(char, '')
    return ' '.join(text.split())

@st.cache_data(ttl=60)  # Cache STT results for 1 minute
def stt(audio_data):
    """Cross-platform speech-to-text conversion"""
    try:
        audio_buffer = io.BytesIO(audio_data)
        sample_rate, audio_array = wavfile.read(audio_buffer)
        
        # Validate audio input
        if np.max(np.abs(audio_array)) < 100:
            return None
            
        # Convert stereo to mono if needed
        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=1)
            
        # Normalize audio
        if audio_array.dtype != np.int16:
            audio_array = (audio_array * 32767).astype(np.int16)

        # Write audio to a temporary WAV file to upload to AssemblyAI
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            temp_path = tf.name
            wavfile.write(temp_path, sample_rate, audio_array)

        try:
            # Upload the file to AssemblyAI (upload endpoint expects binary upload)
            upload_url = "https://api.assemblyai.com/v2/upload"
            with open(temp_path, "rb") as f:
                # Stream upload in chunks to avoid loading entire file into memory
                def gen():
                    while True:
                        chunk = f.read(524288)
                        if not chunk:
                            break
                        yield chunk
                resp = requests.post(upload_url, headers=ASSEMBLYAI_HEADERS, data=gen())
            resp.raise_for_status()
            upload_response = resp.json()
            audio_url = upload_response.get("upload_url")
            if not audio_url:
                raise RuntimeError("AssemblyAI upload did not return upload_url")

            # Create transcript
            transcript_endpoint = "https://api.assemblyai.com/v2/transcript"
            payload = {"audio_url": audio_url}
            tresp = requests.post(transcript_endpoint, headers={**ASSEMBLYAI_HEADERS, "content-type": "application/json"}, json=payload)
            tresp.raise_for_status()
            tid = tresp.json().get("id")
            # Poll for completion
            status = "processing"
            transcript_text = None
            poll_endpoint = f"https://api.assemblyai.com/v2/transcript/{tid}"
            for _ in range(60):  # up to ~60 * 1s = 60s polling
                time.sleep(1)
                presp = requests.get(poll_endpoint, headers=ASSEMBLYAI_HEADERS)
                presp.raise_for_status()
                status = presp.json().get("status")
                if status == "completed":
                    transcript_text = presp.json().get("text")
                    break
                if status == "error":
                    raise RuntimeError(f"AssemblyAI transcription error: {presp.json().get('error')}" )

            return transcript_text.strip() if transcript_text else None
        finally:
            try:
                pathlib.Path(temp_path).unlink(missing_ok=True)
            except:
                pass

    except Exception as e:
        st.error(f"Speech-to-text failed: {str(e)}")
        return None

@st.cache_data(ttl=60)  # Cache TTS results for 1 minute
def tts(text):
    """Convert text to speech with error handling.

    Returns a tuple: (audio_bytes, mime_type) where mime_type is like 'audio/wav' or 'audio/mpeg'.
    Attempts to convert mp3 -> wav using pydub for browser compatibility; falls back to returning mp3 if conversion fails.
    """
    # AssemblyAI does not provide a TTS endpoint. Use the browser SpeechSynthesis fallback instead.
    # Returning None causes the UI renderer to call the browser's Web Speech API with the assistant text.
    return None

def process_message(text):
    """Process a message and generate response"""
    try:
        st.session_state.is_processing = True
        
        with st.spinner("Thinking..."):
            # Add user message
            msg_id = str(uuid.uuid4())
            st.session_state.messages.append({
                "role": "user",
                "content": text,
                "id": msg_id,
                "ts": time.time()
            })
            
            # Generate response
            response = model.generate_content(text)
            response_text = getattr(response, "text", "Sorry, I couldn't generate a response.")
            
            with st.spinner("Generating audio..."):
                # Generate audio if needed. tts() returns (bytes, mime) or None
                audio = tts(response_text) if response_text else None
            
            # Add assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "audio": audio,  # either None or (bytes, mime)
                "id": str(uuid.uuid4()),
                "ts": time.time()
            })
        
        return True
    except Exception as e:
        st.error(f"Failed to process message: {str(e)}")
        return False
    finally:
        st.session_state.is_processing = False

# Add this helper function in the UTILS section
def get_autoplay_audio_html(audio_blob, text=None):
    """Cross-browser compatible audio autoplay.

    - If `audio_blob` is a tuple (bytes, mime) or raw bytes, embed and play that audio.
    - If `audio_blob` is None but `text` is provided, use the browser Web Speech API (speechSynthesis)
      to speak the text locally (no external TTS required).
    """
    # If we have server-generated audio, embed it as before
    if audio_blob:
        if isinstance(audio_blob, tuple) and len(audio_blob) == 2:
            audio_bytes, mime = audio_blob
        else:
            audio_bytes = audio_blob
            mime = "audio/wav"

        b64 = base64.b64encode(audio_bytes).decode()
        # Use the provided mime in the data URI so browser knows how to play it
        return f"""
            <audio autoplay playsinline>
                <source src="data:{mime};base64,{b64}" type="{mime}">
            </audio>
            <script>
                document.querySelectorAll('audio[autoplay]').forEach(audio => {{
                    audio.play().catch(e => console.log("Auto-play prevented:", e));
                }});
            </script>
        """

    # Use espeakng for TTS if text is provided
    if text:
        try:
            import subprocess
            import base64
            import wave
            import io
            import re
            
            # Clean up the text
            def clean_text_for_tts(text):
                # Remove URLs
                text = re.sub(r'http[s]?://\S+', '', text)
                # Remove markdown code blocks and inline code
                text = re.sub(r'`[^`]+`', '', text)
                text = re.sub(r'```[\s\S]+?```', '', text)
                # Remove special characters but keep basic punctuation
                text = re.sub(r'[^a-zA-Z0-9\s.,!?()-]', ' ', text)
                # Replace multiple spaces with single space
                text = re.sub(r'\s+', ' ', text)
                # Replace multiple periods with single period
                text = re.sub(r'\.{2,}', '.', text)
                # Add spaces after punctuation if missing
                text = re.sub(r'([.,!?])([a-zA-Z])', r'\1 \2', text)
                return text.strip()
            
            # Clean the text
            cleaned_text = clean_text_for_tts(text)
            if not cleaned_text:
                return None
                
            # Use espeak-ng directly to generate wav data
            cmd = ['espeak-ng', '-w', '/dev/stdout', '-v', 'en-us', cleaned_text]
            process = subprocess.run(cmd, capture_output=True, check=True)
            raw_audio = process.stdout
            
            # Create WAV data in memory
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 2 bytes per sample
                wav_file.setframerate(22050)  # Standard sample rate
                wav_file.writeframes(raw_audio)
            
            # Get the WAV data
            wav_data = wav_buffer.getvalue()
            
            # Convert to base64 for HTML audio
            b64_audio = base64.b64encode(wav_data).decode()
            audio_src = f"data:audio/wav;base64,{b64_audio}"
            
            # Return audio element that autoplays
            return f'''
                <audio autoplay style="display:none">
                    <source src="{audio_src}" type="audio/wav">
                </audio>
            '''
        except Exception as e:
            st.error(f"TTS Error: {str(e)}")
            return None

    return ""

# Add this helper function near the top of the file after imports
def safe_rerun():
    """Safe rerun helper for different Streamlit versions"""
    try:
        st.rerun()
    except AttributeError:
        try:
            st.experimental_rerun()
        except:
            st.empty()  # Fallback - clear cache and continue

# Add to UTILS section:
def retry_with_backoff(func, max_retries=3):
    """Retry a function with exponential backoff"""
    for i in range(max_retries):
        try:
            return func()
        except Exception as e:
            if i == max_retries - 1:
                raise e 
            time.sleep(2 ** i)  # exponential backoff

# ---------- UI LAYOUT ----------
st.markdown("""
<style>
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
}

.user-message {
    background-color: #2557a7;
    margin-left: 20%;
}

.assistant-message {
    background-color: #1e1e1e;
    margin-right: 20%;
}

.message-content {
    color: white;
}

.audio-player {
    opacity: 0.8;
    height: 25px;
    margin-top: 0.5rem;
}

audio {
    max-width: 100%;
    min-width: 200px;
}

audio::-webkit-media-controls-panel,
audio::-moz-media-controls-panel,
audio::-ms-media-controls-panel {
    background-color: rgba(255, 255, 255, 0.1);
}

/* Hide audio player when it's autoplaying but keep it accessible */
audio[autoplay] {
    height: 0;
    width: 0;
    opacity: 0;
    position: absolute;
}

.input-area {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: #0e1117;
    padding: 1rem;
    display: flex;
    gap: 1rem;
    align-items: center;
}

/* Cross-browser audio player styles */
audio[controls] {
    display: block;
    margin: 0.5rem 0;
}

/* Progressive enhancement for modern browsers */
@supports (backdrop-filter: blur(10px)) {
    .chat-message {
        backdrop-filter: blur(10px);
    }
}
</style>
""", unsafe_allow_html=True)

# Add to the UI section before the chat history:
st.markdown("""
<script>
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + Enter to send
    if ((e.ctrlKey || e.metaKey) && e.keyCode === 13) {
        document.querySelector('button[type="submit"]').click();
    }
    // Esc to clear
    if (e.keyCode === 27) {
        document.querySelector('button[aria-label="Clear chat history"]').click();
    }
});
</script>
""", unsafe_allow_html=True)

# Chat history
st.title("💬 Gemini Voice Assistant")
for msg in st.session_state.messages:
    message_type = "user-message" if msg["role"] == "user" else "assistant-message"
    if msg["role"] == "assistant":
        # Pass audio tuple if present, otherwise pass text so browser TTS can speak
        audio_html = get_autoplay_audio_html(msg.get("audio"), text=msg.get("content"))
    else:
        audio_html = ""

    st.markdown(f"""
        <div class="chat-message {message_type}">
            <div class="message-content">{msg["content"]}</div>
        </div>
    """, unsafe_allow_html=True)

    # Render audio/script with enough space to show debug output
    if audio_html:
        try:
            # Use larger height to show debug messages
            components.html(audio_html, height=100)
        except Exception as e:
            st.error(f"Audio playback failed: {str(e)}")

# Input area
with st.container():
    cols = st.columns([6, 2, 1])
    
    # Text input with form
    with cols[0]:
        with st.form(key="message_form", clear_on_submit=True):
            user_input = st.text_input("Message:", key="user_input")
            submit_button = st.form_submit_button("Send")
            if submit_button and user_input:
                process_message(user_input)
                safe_rerun()
    
    # Voice input
    with cols[1]:
        audio_input = st.audio_input("Speak", key=f"audio_{st.session_state.audio_key}")
        if audio_input:
            audio_bytes = audio_input.getvalue()
            if st.session_state.last_audio != audio_bytes:
                st.session_state.last_audio = audio_bytes
                st.session_state.audio_key += 1
                if text := stt(audio_bytes):
                    process_message(text)
                    safe_rerun()
    
    # Clear chat button
    with cols[2]:
        if st.button("Clear", help="Clear chat history"):
            st.session_state.messages = []
            safe_rerun()

# Auto-scroll to bottom
st.markdown("""
<script>
    const messages = document.querySelector('.main');
    if (messages) {
        messages.scrollTop = messages.scrollHeight;
    }
</script>
""", unsafe_allow_html=True)