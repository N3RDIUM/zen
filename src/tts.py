from pydub import AudioSegment
from pydub.playback import play
from os import remove
import soundfile as sf
from tts_helper import load_text_to_speech, timer, load_voice_style

# settings
total_step = 24
speed = 1.05
voice_style_paths = ["assets/voice_styles/M1.json"]
batch = True
onnx_dir = "assets/onnx"
use_gpu = False

bsz = len(voice_style_paths)
text_to_speech = load_text_to_speech(onnx_dir, use_gpu)
style = load_voice_style(voice_style_paths, verbose=True)

def speak(text):
    with timer("Generating speech from text"):
        wav, duration = text_to_speech(text, style, total_step, speed)

    for b in range(bsz):
        fname = "result.wav"
        w = wav[b, : int(text_to_speech.sample_rate * duration[b].item())]  # [T_trim]
        sf.write(fname, w, text_to_speech.sample_rate)

    song = AudioSegment.from_wav("result.wav")
    play(song)
    remove("result.wav")

