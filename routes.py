from flask import Flask, render_template, Response, stream_with_context, request, jsonify
from threading import Thread, Event, Lock
import cv2
import pyaudio
import wave
import openai
from config import OPENAI_API_KEY
from audio_transcription import transcribe_audio, callWhisper,generate_summary,update_transcription
from question_answer import query_openai, load_documents
from Quiz import get_mca_questions,get_keywords


app = Flask(__name__)

# Load OpenAI API key
openai.api_key = OPENAI_API_KEY

client = openai.OpenAI(api_key=OPENAI_API_KEY)

camera = cv2.VideoCapture(0)

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
# Define transcriptions as a global variable


# Global variables
transcriptions = {}
waiting = False

# Initialize a lock
transcriptions_lock = Lock()

@app.route('/')
def index():
    return render_template('UI.html', transcription='')

@app.route('/video')
def video():
    return Response(stream_with_context(generate_frames()), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start-transcription')
def start_transcription():
    global waiting
    if not waiting:
        waiting = True
        stop_event.clear()
        t = Thread(target=transcribe_audio, args=(stop_event,))
        t.start()
    return "Waiting for transcription..."

@app.route('/stop-transcription')
def stop_transcription():
    stop_event.set()
    return "Stopped transcription."



@app.route('/qa-page')
def qa_page():
    return render_template('UI2.html', transcriptions=transcriptions)

@app.route('/answer-question', methods=['POST'])
def answer_question():
    data = request.json
    question = data['question']
    documents = load_documents("./transcriptions.txt")  # Load documents from the file
    answer = query_openai(question, documents)
    return jsonify({'answer': answer})

@app.route('/save-transcriptions', methods=['POST'])
def save_transcriptions():
    data = request.json
    transcription_text = data['transcriptionText']
    with open("transcriptions.txt", "w", encoding="utf-8") as file:
        file.write(transcription_text)
    return jsonify({'message': 'Transcriptions saved successfully'})

@app.route('/summarize-transcriptions', methods=['POST'])
def summarize_transcriptions():
    # Get the transcript content from the request body
    transcript = request.json['transcript']

    # Use GPT-3 API to generate a summary
    summary = generate_summary(transcript)
    
    return summary
@app.route('/get-transcription')
def get_transcription():
    result = update_transcription()
    return result
    

@app.route('/generate-quiz', methods=['POST'])
def generate_quiz():
    try:
        content = request.json['text']  # This matches the JSON key sent from the frontend
        questions = get_mca_questions(content)  # Assume this returns a list of strings
        print(questions)
        return questions, 200, {'Content-Type': 'application/json'} # Return questions as JSON
    except Exception as e:
        return jsonify({"error": str(e)}), 400

#@app.route('/generate-keywords', methods=['POST'])
#def generate_quiz():
#    try:
#        content = request.json['text']  # This matches the JSON key sent from the frontend
#        questions = get_keywords(content)  # Assume this returns a list of strings
#        print(questions)
#        return questions, 200, {'Content-Type': 'application/json'} # Return questions as JSON
#    except Exception as e:
#        return jsonify({"error": str(e)}), 400

@app.route('/ui3')
def ui3():
    return render_template('UI3.html')

if __name__ == "__main__":
    app.run(debug=True)
