# audio_transcription.py
import pyaudio
import wave
import cv2
import openai
from config import OPENAI_API_KEY
from threading import Thread, Event,Lock
from openai import OpenAI


# Load OpenAI API key
openai.api_key = OPENAI_API_KEY

client = openai.OpenAI(api_key=OPENAI_API_KEY)

camera = cv2.VideoCapture(0)
transcriptions_lock = Lock()

# Set up audio recording
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 10
transcriptions = {}
transcription_count = -1
waiting = False

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

stop_event = Event()

def transcribe_audio(stop_event):
    global transcription_count
    while not stop_event.is_set():
        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            if not stop_event.is_set():
                data = stream.read(CHUNK)
                frames.append(data)

        # Write the audio to file
        wf = wave.open("audio.wav", 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        t = Thread(target=callWhisper)
        t.start()

def generate_summary(text):
    # Create a prompt for the chat completion
    prompt_text = f"Please do a detailed summary the following text:\n{text}\n\nSummary:"

    # Call OpenAI's Chat Completion API to generate a summary
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_text}
        ]
    )

    # Extract the summary from the response
    summary = response.choices[0].message.content
    
    return summary

def callWhisper():
    global transcription_count, transcriptions
    with open("audio.wav", "rb") as audio_file:
        transcription_count += 1
        local_count = transcription_count
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="ta",
            response_format="text"
        )
        # Acquire the lock before updating transcriptions
        with transcriptions_lock:
            transcriptions[local_count] = transcription
        print(f"Transcribed audio from segment {local_count}: ")
        print(transcription)
    print(transcriptions)
    
@app.route('/get-transcription')
def get_transcription():
    global waiting, transcriptions
    print("Content is \n")
    # Acquire the lock before accessing transcriptions
    with transcriptions_lock:
        print(transcriptions)
        content = ''
        if transcriptions:
            for key, value in transcriptions.items():
                content += f"Transcription {key}: {value}\n"
            waiting = False
        elif not waiting:
            content = "Waiting for transcription..."
        print("Content is \n")
        print(content)
    return content
