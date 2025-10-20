from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai
import os
import pvleopard
import pvorca
import struct
import wave
import tempfile
import io
import numpy as np
from scipy.io import wavfile
import base64
import time
import uuid
import platform
import pathlib

load_dotenv()

# ---------- CONFIG ----------
st.set_page_config(page_title="Gemini Voice Assistant", page_icon="🎤", layout="wide")

# Initialize APIs with error handling
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-2.5-flash', 
                                system_instruction="Keep responses concise and professional.")
    
    picovoice_access_key = os.environ["PICOVOICE_ACCESS_KEY"]
    leopard = pvleopard.create(access_key=picovoice_access_key)
    orca = pvorca.create(access_key=picovoice_access_key)
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

        # Cross-platform temp directory handling
        if platform.system() == 'Windows':
            temp_dir = pathlib.Path.home() / 'AppData' / 'Local' / 'Temp'
        else:
            temp_dir = pathlib.Path('/tmp')

        # Ensure temp directory exists
        temp_dir.mkdir(parents=True, exist_ok=True)
            
        # Use pathlib for cross-platform path handling
        temp_path = temp_dir / f"temp_audio_{uuid.uuid4()}.wav"
        
        try:
            # Write and process audio file
            wavfile.write(str(temp_path), sample_rate, audio_array)
            transcript, _ = leopard.process_file(str(temp_path))
            return transcript.strip() if transcript else None
        finally:
            # Clean up temp file in all cases
            try:
                temp_path.unlink(missing_ok=True)  # Cross-platform file deletion
            except:
                pass

    except Exception as e:
        st.error(f"Speech-to-text failed: {str(e)}")
        return None

@st.cache_data(ttl=60)  # Cache TTS results for 1 minute
def tts(text):
    """Convert text to speech with error handling"""
    try:
        pcm, _ = orca.synthesize(clean_text(text))
        audio_buffer = io.BytesIO()
        with wave.open(audio_buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(orca.sample_rate)
            wav.writeframes(struct.pack(f"{len(pcm)}h", *pcm))
        return audio_buffer.getvalue()
    except Exception as e:
        st.error(f"Text-to-speech failed: {str(e)}")
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
                # Generate audio if needed
                audio = tts(response_text) if response_text else None
            
            # Add assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "audio": audio,
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
def get_autoplay_audio_html(audio_bytes):
    """Cross-browser compatible audio autoplay"""
    if not audio_bytes:
        return ""
    b64 = base64.b64encode(audio_bytes).decode()
    return f"""
        <audio autoplay playsinline>
            <source src="data:audio/wav;base64,{b64}" type="audio/wav">
            <source src="data:audio/wav;base64,{b64}" type="audio/x-wav">
            <source src="data:audio/wav;base64,{b64}" type="audio/wave">
        </audio>
        <script>
            document.querySelectorAll('audio[autoplay]').forEach(audio => {{
                audio.play().catch(e => console.log("Auto-play prevented:", e));
            }});
        </script>
    """

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
    audio_html = get_autoplay_audio_html(msg.get("audio")) if msg["role"] == "assistant" else ""
    
    st.markdown(f"""
        <div class="chat-message {message_type}">
            <div class="message-content">{msg["content"]}</div>
            {audio_html}
        </div>
    """, unsafe_allow_html=True)

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