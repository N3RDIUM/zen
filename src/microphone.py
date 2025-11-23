import logging
import os
import time

import speech_recognition as sr
import torch
from faster_whisper import WhisperModel

from logger import logger

MODEL = "base.en"
logger.log(
    logging.DEBUG,
    f"DEBUG [ whyspr    ]  Loading OpenAI Whisper model `{MODEL}`...",
)
model = WhisperModel(MODEL, device="cpu", compute_type="int8")
pipeline = model.transcribe
logger.log(
    logging.DEBUG,
    f"DEBUG [ whyspr    ]  Model `{MODEL}` loaded successfully!",
)

r = sr.Recognizer()
m = sr.Microphone()
with m as source:
    logger.log(
        logging.INFO,
        "DEBUG [ whyspr    ]  Adjusting for ambient noise...",
    )
    r.adjust_for_ambient_noise(source)

def rec_from_mic():
    with m as source:
        logger.log(
            logging.INFO,
            "DEBUG [ whyspr    ]  Listening...",
        )
        audio = r.listen(source)
        logger.log(
            logging.INFO,
            "DEBUG [ whyspr    ]  Audio capture complete! Recognizing...",
        )

    with open("audio.wav", "wb") as f:
        f.write(audio.get_wav_data())

    t = time.perf_counter()
    with torch.inference_mode():
        segments, info = pipeline("audio.wav", vad_filter=True)
        logger.log(
            logging.INFO,
            "DEBUG [ whyspr    ]  Detected language '%s' with probability %f"
            % (info.language, info.language_probability),
        )

    recognized = ""
    for segment in segments:
        logger.log(
            logging.INFO,
            "DEBUG [ whyspr    ]  [%.2fs -> %.2fs] %s"
            % (segment.start, segment.end, segment.text),
        )
        recognized += segment.text

    logger.log(
        logging.INFO,
        f"DEBUG [ whyspr    ]  Transcription took {time.perf_counter() - t} seconds!",
    )
    os.remove("audio.wav")

    return recognized

