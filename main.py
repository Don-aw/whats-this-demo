import pyaudio
import vosk
import torch
from PIL import Image
import cv2
import time

from lavis.models import load_model_and_preprocess

# Initialize Vosk ASR model
# This requires a downloaded vosk model at the same directory with main.py
model = vosk.Model("vosk-model-small-en-us-0.15")
rec = vosk.KaldiRecognizer(model,
                           16000,
                           '["what is this", "tell me about it", "how does it work", "[unk]"]')

# Audio setup
# This section requires the default microphone of the device to be set in the system
audio = pyaudio.PyAudio()

# Setup CPU/GPU to use for image-text
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, vis_processors, txt_processors = \
    load_model_and_preprocess(name="blip_vqa", model_type="vqav2", is_eval=True, device=device)


def close_stream():
    stream.close()


def open_stream():
    stream = audio.open(
        rate=16000, channels=1, format=pyaudio.paInt16, input=True, frames_per_buffer=8000
    )


def inquire():

    print("inquire")

    # raw_image = Image.open("vqa_demo.png").convert("RGB")
    # Initialize video camera for taking photos
    cam = cv2.VideoCapture(0)
    _, cv_img = cam.read()
    rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    raw_image = Image.fromarray(rgb_img).convert("RGB")
    raw_image.show()
    cam.release()

    question = "What's the thing held by the person?"

    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    question = txt_processors["eval"](question)

    samples = {"image": image, "text_input": question}

    print(model.predict_answers(samples=samples, inference_method="generate"))

def history():
    print("history")


def explain():
    print("explain")


# Idle loop, runs audio detection
stream = audio.open(
    rate=16000, channels=1, format=pyaudio.paInt16, input=True, frames_per_buffer=8000
)
print("Listening for wake words...")

while True:
    # False stops exception for overflowing buffer (due to async function calling)
    data = stream.read(4000, False)

    # When a sentence is finished (there is some amount of silence after voice)
    if rec.AcceptWaveform(data):
        result = rec.Result()
        text = result.strip().lower()

        # Looking for exact matches for any of the wake words in a sentence
        if "what is this" in text:
            inquire()
        elif "tell me about" in text:
            history()
        elif "how does it work" in text:
            explain()

    # Debugging purposes, to check if partials are correctly recognizing the speech
    # else:
        # partial = rec.PartialResult()
        # print(partial)
