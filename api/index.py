from flask import Flask, request, render_template, jsonify
import openai
import pygame
import io
import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
import tempfile
import os
from functools import wraps
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tts_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app with configuration
app = Flask(__name__)
app.config.update(
    MAX_CONTENT_LENGTH=16 * 1024,  # Limit text input to 16KB
    SUPPORTED_VOICES=['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'],
    DEFAULT_VOICE='alloy',
    DEFAULT_MODEL='tts-1',
)

# Initialize OpenAI client
client = openai.OpenAI()

def init_mixer():
    """Initialize pygame mixer with error handling"""
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=24000)
            logger.info("Pygame mixer initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pygame mixer: {str(e)}")
        raise

class AudioSynthesisError(Exception):
    """Custom exception for audio synthesis errors"""
    pass

def error_handler(f):
    """Decorator for consistent error handling"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except AudioSynthesisError as e:
            logger.error(f"Audio synthesis error: {str(e)}")
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return jsonify({'error': 'An unexpected error occurred'}), 500
    return decorated_function

def analyze_audio_amplitude(audio_content):
    """Analyze audio content and return amplitude data for animation"""
    # Convert MP3 to WAV for analysis
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_mp3:
        temp_mp3.write(audio_content)
        temp_mp3_path = temp_mp3.name

    audio = AudioSegment.from_mp3(temp_mp3_path)
    
    # Export as WAV for numpy analysis
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
        audio.export(temp_wav.name, format='wav')
        temp_wav_path = temp_wav.name

    # Read WAV file
    sample_rate, audio_data = wavfile.read(temp_wav_path)
    
    # Clean up temporary files
    os.unlink(temp_mp3_path)
    os.unlink(temp_wav_path)

    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    # Calculate amplitude envelope
    frame_size = int(sample_rate / 30)  # 30 fps
    amplitude_envelope = []
    
    for i in range(0, len(audio_data), frame_size):
        chunk = audio_data[i:i + frame_size]
        amplitude = np.abs(chunk).mean()
        amplitude_envelope.append(float(amplitude))

    # Normalize amplitudes to range [0, 1]
    max_amp = max(amplitude_envelope)
    if max_amp > 0:
        amplitude_envelope = [amp / max_amp for amp in amplitude_envelope]

    return amplitude_envelope

def synthesize_and_play_audio(text, voice=None):
    """Synthesize and play audio with amplitude analysis"""
    try:
        voice = voice if voice in app.config['SUPPORTED_VOICES'] else app.config['DEFAULT_VOICE']
        
        # Request audio synthesis
        response = client.audio.speech.create(
            model=app.config['DEFAULT_MODEL'],
            voice=voice,
            input=text
        )
        audio_content = response.content
        
        # Analyze audio for animation
        amplitude_data = analyze_audio_amplitude(audio_content)
        
        # Play the audio content
        init_mixer()
        sound = pygame.mixer.Sound(io.BytesIO(audio_content))
        sound.play()
        
        logger.info(f"Successfully synthesized and analyzed audio for text: {text[:50]}...")
        return amplitude_data
        
    except Exception as e:
        logger.error(f"Failed to process audio: {str(e)}")
        raise AudioSynthesisError(f"Failed to process audio: {str(e)}")

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html', 
                         supported_voices=app.config['SUPPORTED_VOICES'],
                         default_voice=app.config['DEFAULT_VOICE'])

@app.route('/synthesize', methods=['POST'])
@error_handler
def synthesize():
    """Handle text-to-speech synthesis requests with animation data"""
    data = request.get_json()
    if not data or 'text' not in data:
        raise AudioSynthesisError("No text provided")
    
    text = data['text'].strip()
    voice = data.get('voice', app.config['DEFAULT_VOICE'])
    
    if not text:
        raise AudioSynthesisError("Empty text provided")
    
    if len(text) > app.config['MAX_CONTENT_LENGTH']:
        raise AudioSynthesisError("Text exceeds maximum length")
    
    amplitude_data = synthesize_and_play_audio(text, voice)
    return jsonify({
        'status': 'success',
        'message': 'Audio synthesized and played successfully',
        'text_length': len(text),
        'voice': voice,
        'amplitude_data': amplitude_data
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'pygame_mixer': pygame.mixer.get_init() is not None,
        'supported_voices': app.config['SUPPORTED_VOICES']
    })

if __name__ == '__main__':
    app.run(debug=True)
