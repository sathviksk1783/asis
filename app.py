import os
import json
import wave
import pyttsx3
import pyaudio
from vosk import Model, KaldiRecognizer
from threading import Thread
from flask import Flask, request, render_template, jsonify
import google.generativeai as genai
import pdfplumber

# Initialize Flask App
app = Flask(__name__)

# Initialize Gemini API Key
genai.configure(api_key="AIzaSyBGgt0we-tJSgd5G8wUUDNUC7fIsf9Lo-k")  # Replace with your actual key

# Speech recognition variables
recognizer = None
model = None
stream = None
is_recording = False
full_text = ""  # Store the entire transcript
latest_input = ""  # Store the latest user input

# Load Vosk Model
def load_vosk_model():
    model_path = 'vosk-model-en-in-0.5'  # Ensure this path is correct
    if not os.path.exists(model_path):
        raise Exception(f"Vosk model not found at {model_path}")
    return Model(model_path)

# Function to start recording
def start_recording():
    global recognizer, stream, is_recording, full_text, latest_input, model
    try:
        # Ensure model is loaded
        if not model:
            model = load_vosk_model()
        
        recognizer = KaldiRecognizer(model, 16000)
        speech = pyaudio.PyAudio()
        stream = speech.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4096)
        recognizer.Reset()
        full_text = ""
        is_recording = True

        while is_recording:
            data = stream.read(4096, exception_on_overflow=False)
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").strip()
                if text:
                    full_text += text + " "
                    latest_input = text
                    print(f"USER INPUT: {full_text}")
    except Exception as e:
        print(f"Recording error: {e}")
        is_recording = False

# Function to stop recording
def stop_recording():
    global is_recording, stream
    is_recording = False
    
    # Stop the Vosk stream but leave the text for further processing
    if stream:
        stream.stop_stream()
        stream.close()
        stream = None

    # Return the full text captured during the recording
    return full_text

# Text-to-Speech initialization
engine = pyttsx3.init()

def read_aloud(text):
    engine.say(text)
    engine.runAndWait()

# Function to extract text from PDF
def extract_pdf_text(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ''.join(page.extract_text() for page in pdf.pages)
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

# Function to process files with Gemini API
# def gemini_file_task(file_path, prompt):
#     try:
#         mime_type = "application/pdf"
#         uploaded_file = genai.upload_file(file_path, mime_type=mime_type)
#         file_name = uploaded_file.name
#         print(f"File uploaded: {file_name}")

#         retrieved_file = genai.get_file(file_name)
#         print(f"Retrieved File: {retrieved_file}")

#         model = genai.GenerativeModel("gemini-1.5-flash")
#         result = model.generate_content([retrieved_file, prompt])
#         return result.text
#     except Exception as e:
#         print(f"Error processing file: {e}")
#         return None


def gemini_file_task(file_path=None, prompt=None):
    try:
        if file_path:  # File-based processing
            mime_type = "application/pdf"
            uploaded_file = genai.upload_file(file_path, mime_type=mime_type)
            file_name = uploaded_file.name
            print(f"File uploaded: {file_name}")
            
            retrieved_file = genai.get_file(file_name)
            print(f"Retrieved File: {retrieved_file}")

            model = genai.GenerativeModel("gemini-1.5-flash")
            result = model.generate_content([retrieved_file, prompt])
            return result.text
        
        elif prompt:  # Text-based processing
            model = genai.GenerativeModel("gemini-1.5-flash")
            result = model.generate_content([prompt])
            return result.text
        else:
            raise ValueError("Either file_path or prompt must be provided.")

    except Exception as e:
        print(f"Error processing request: {e}")
        return None





# Process transcription with model
def process_transcription_with_model(transcription):
    prompt = f"You are a helpful assistant. Here is the text: '{transcription}'. Please provide a concise and clear response to this text."
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content([transcription, prompt])
    return response


def process_transcription_with_model(transcription):
    print(f"Received transcription: {transcription}")  # Debugging: Ensure transcription is received

    # Replace this logic with your actual model call
    class GenerateContentResponse:
        def __init__(self, transcription):
            self.transcription = transcription  # Mock processing for now
            
        def to_dict(self):
            return {
            "transcription": self.transcription
        }
    
    response = GenerateContentResponse(transcription)
    print(f"Response from model: {response.to_dict()}")
    print(response)  # Debugging: See what the response looks like
    return response




# Templates for prompts
pdf_summary_template = """
Your task is to carefully read and thoroughly understand the attached PDF document. Then, craft a summary that is formal, accurate, and well-structured. Ensure that your response is clear, concise, and professional, providing a comprehensive overview of the content while maintaining precision.

Include the following elements in your response:

1. Title and Context

Begin by identifying the title of the document and providing an overview of its purpose and subject matter.
Set the stage for the reader by explaining the document's focus and relevance.
2. Key Sections

Break the document into its primary sections, summarizing each one in a clear and organized manner.
Highlight the central ideas, findings, and arguments without unnecessary embellishments.
3. Important Definitions and Concepts

Clearly define any technical terms or specialized concepts presented in the document.
Provide straightforward explanations to ensure the summary remains accessible and informative.
4. Key Insights

Focus on the major conclusions or findings of the document.
Explain why these insights are significant and their implications within the relevant context.
5. Contextual Relevance

Discuss the importance of the document, including its potential applications or impact.
Explain why the content is relevant to the intended audience or field of study.
6. Length and Detail

The summary should include sufficient detail to accurately reflect the content of the document while being concise and to the point.
Aim for a formal tone and structured flow, avoiding casual language or filler words.
Your response should be professional, precise, and reflective of the document's contents, ensuring clarity and coherence throughout.

Use the following content:
{pdf_text}
"""

podcast_summary_template = """
You are a friendly and engaging podcast creator with a talent for breaking down complex ideas into relatable and conversational stories. Your task is to read and understand the attached PDF document, capturing its content and intent in a clear and approachable way.

Using the PDF content, create a lively and engaging podcast episode. The goal is to make listeners feel like they are having a friendly chat with you over coffee. You don't need to introduce any outside context—just focus on the document's material. Use natural pauses, casual phrases, and a conversational tone throughout.

Include the following elements in your response:

Introduction:
- Start by introducing the document's main themes and ideas.
- Highlight any notable points, such as its purpose and key takeaways.
- Use informal phrases like “Alright, so,” or “You know what's interesting about this?” to make it more relatable.

Detailed Breakdown:
- Go through the document's content in a simple, engaging way.
- Use analogies and examples to explain key points, like “It's kind of like when you…” or “Imagine this…”
- Use conversational filler words like, “Hmm, let me think,” or “Oh, this is a good one,” to create a natural flow.

Contextual Insights:
- Discuss how the document's ideas connect to the real world.
- Use transitions like, “Okay, so here's the thing,” or “This is where it gets really interesting” to keep the conversation flowing.

Conclusions and Takeaways:
- Wrap up the discussion by summarizing the main points.
- Pose thought-provoking questions like, “Have you ever wondered why this happens?” to engage listeners.

Engaging Storytelling:
- Make it a two person conversation and not one single person talking everything.
- Use storytelling techniques to make the document's points memorable.
- Phrases like “Alright, picture this…” or “Let me tell you a quick story about…” will bring everything to life.

{pdf_text}
"""

# Routes for Flask
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/voice_to_text', methods=['GET', 'POST'])
def voice_to_text():
    global is_recording, full_text
    if request.method == 'POST':
        action = request.form.get('action')
        
        if action == 'start':
            # Start recording in a separate thread
            thread = Thread(target=start_recording, daemon=True)
            thread.start()
            return jsonify({"status": "recording started"})
        
        elif action == 'stop':
            # Stop recording and get the transcription
            transcription = stop_recording()
            # transcription="what is e equal to m c squared"
            print(f"Transcription received after stop: {transcription}")

            # Process the transcription directly with the Gemini model
            prompt = f"Respond to the following transcription:\n\n{transcription}"
            response = gemini_file_task(prompt=prompt)  # Text-based input
            
            # Return the transcription and the response
            return jsonify({
                "transcription": transcription,
                "response": response
            })
    
    return render_template('speak.html')


@app.route('/pdf_summary', methods=['GET', 'POST'])
def pdf_summary():
    if request.method == 'POST':
        pdf_file = request.files['pdf_file']
        if pdf_file:
            upload_dir = "uploads"
            os.makedirs(upload_dir, exist_ok=True)
            file_path = os.path.join(upload_dir, pdf_file.filename)
            pdf_file.save(file_path)
            pdf_text = extract_pdf_text(file_path)
            if pdf_text:
                prompt = pdf_summary_template
                summary = gemini_file_task(file_path, prompt)
                return jsonify({"summary": summary})
    return render_template('pdf_summary.html')


@app.route('/podcast_summary', methods=['GET', 'POST'])
def podcast_generator():
    if request.method == 'POST':
        pdf_file = request.files['pdf_file']
        if pdf_file:
            upload_dir = "uploads"
            os.makedirs(upload_dir, exist_ok=True)
            file_path = os.path.join(upload_dir, pdf_file.filename)
            pdf_file.save(file_path)
            pdf_text = extract_pdf_text(file_path)  # Extract text from PDF
            if pdf_text:
                prompt = podcast_summary_template
                podcast_content = gemini_file_task(file_path, prompt)  
                print(podcast_content)
                return jsonify({"podcast_content": podcast_content})
    return render_template('podcast_summary.html')


@app.route('/read_aloud', methods=['POST'])
def read_aloud_content():
    text = request.json.get('text')
    if text:
        read_aloud(text)
        return jsonify({"status": "reading started"})
    return jsonify({"status": "no text provided"})


if __name__ == '__main__':
    app.run(debug=True)
    