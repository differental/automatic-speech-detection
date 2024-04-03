# apt install portaudio19-dev
# pip install accelerate wave flask flask_socketio flask_cors pyaudio torch transformers
# Might need restart after installing accelerate
import wave
import threading
from flask import Flask, render_template
from flask_socketio import SocketIO
from flask_cors import CORS
import pyaudio
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "distil-whisper/distil-medium.en"

model_whisper = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model_whisper.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model_whisper,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    torch_dtype=torch_dtype,
    device=device,
)

model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=False,
                              onnx=False)
(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils
vad_iterator = VADIterator(model)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
CORS(app)
socketio = SocketIO(app)

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1536

# 1 frame: 1024/44100=23ms
# 10 frames together, send in 5 frame intervals (1-10, 6-15, 11-20...)

frames5 = []
frames10 = []
all_frames = []
all_speech_probs = []
tot_len = 0
recording = False
BOUNDARY = 0.3  # needs adjusting per device/scenario

final_result = ""
calc_result_running = False
fallback = False

my_thread = None

def calc_result(counts):
    global calc_result_running, final_result

    calc_result_running = True

    filename = "test_" + str(counts) + ".wav"
    result = pipe(filename)

    final_result = result["text"]

    calc_result_running = False


@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('start_recording')
def start_recording():
    print("Recording Started")
    global frames5, frames10, all_frames, recording, tot_len
    frames5 = []  # Clear existing frames
    frames10 = []
    all_frames = []
    recording = True
    tot_len = 0

@socketio.on('audio_data')
def handle_audio_data(data):
    global frames5, frames10, all_frames, all_speech_probs, fallback, tot_len, my_thread
    #print('Received audio data length:', len(data))
    socketio.emit('audio_received')
    all_frames.append(data)
    # frames5.append(data) #frames5: 6-15, 16-25, 26-35, 36-45, ...
    frames10.append(data)  # frames10: 1-10, 11-20, 21-30, 31-40, ...


    # Update: Now, one frame is approximately 250ms...
    # frames5: 2-3, 4-5, 6-7...
    # frames10: 1-2, 3-4, 5-6...

    tot_len += 1
    local_len = tot_len

    if local_len >= 2:
        frames5.append(data)

    if local_len % 4 == 0 and not fallback:
        filename = "test_" + str(local_len) + ".wav"
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(all_frames))
        wf.close()
        if calc_result_running and my_thread is not None:
            if my_thread.is_alive():
                fallback = True
                print("Falling back to processing after finish.")
        if not fallback:
            my_thread = threading.Thread(
                target=calc_result, args=(local_len,))
            # print("Running new thread")
            my_thread.start()

    # 200-250ms intervals fed into vad is usually good
    if len(frames5) == 2:
        frames5_data = b''.join(frames5)
        numbers = [int.from_bytes(frames5_data[i:i+2], byteorder='little', signed=False)
                    for i in range(0, len(frames5_data), 2)]

        # Normalize integers to range between 0 and 2
        max_val = max(numbers)
        min_val = min(numbers)
        normalized_numbers = [(num - min_val) * 2 /
                                (max_val - min_val) for num in numbers]

        wav = torch.tensor(normalized_numbers, dtype=torch.float32)

        window_size_samples = 1536
        speech_probs = []
        for i in range(0, len(wav), window_size_samples):
            chunk = wav[i: i + window_size_samples]
            if len(chunk) < window_size_samples:
                break
            speech_prob = model(chunk, 16000).item()
            speech_probs.append(speech_prob)
        vad_iterator.reset_states()
        all_speech_probs += speech_probs[:10]

        print(speech_probs)
        print(max(speech_probs[:10]))

        if max(speech_probs[:10]) <= BOUNDARY:
            socketio.emit('stop_recording')
            print("Stop recording called")
            #stop_recording()

        frames5 = []

    if len(frames10) == 2:
        frames10_data = b''.join(frames10)
        numbers = [int.from_bytes(frames10_data[i:i+2], byteorder='little', signed=False)
                    for i in range(0, len(frames10_data), 2)]

        # Normalize integers to range between 0 and 2
        max_val = max(numbers)
        min_val = min(numbers)
        normalized_numbers = [(num - min_val) * 2 /
                                (max_val - min_val) for num in numbers]

        wav = torch.tensor(normalized_numbers, dtype=torch.float32)

        window_size_samples = 1536
        speech_probs = []
        for i in range(0, len(wav), window_size_samples):
            chunk = wav[i: i + window_size_samples]
            if len(chunk) < window_size_samples:
                break
            speech_prob = model(chunk, 16000).item()
            speech_probs.append(speech_prob)
        vad_iterator.reset_states()
        all_speech_probs += speech_probs[:10]

        print(max(speech_probs[:10]))

        if max(speech_probs[:10]) <= BOUNDARY:
            socketio.emit('stop_recording')
            print("Stop recording called")
            #stop_recording()

        frames10 = []

@socketio.on('stop_recording')
def stop_recording():
    global recording
    recording = False
    print("Recording stopped")

    if not fallback:
        while calc_result_running:
            continue
        print(final_result)
        socketio.emit('transcription_available', final_result)

    else:
        filename = 'recording_all.wav'
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(all_frames))
        wf.close()
        result = pipe("recording_all.wav")
        print(result['text'])
        socketio.emit('transcription_available', result['text'])

if __name__ == '__main__':
    # For testing only, self-signed certificates
    socketio.run(app, ssl_context=("cert.pem", "key.pem"), debug=True, host='0.0.0.0')
