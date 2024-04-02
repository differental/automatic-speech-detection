from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import pyaudio
import wave
import threading
import torch
import numpy as np
import ctypes

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
                              force_reload=True,
                              onnx=False)
(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils
vad_iterator = VADIterator(model)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1536

#1 frame: 1024/44100=23ms
#10 frames together, send in 5 frame intervals (1-10, 6-15, 11-20...)

frames5 = []
frames10 = []
all_frames = []
all_speech_probs = []
recording = False
BOUNDARY = 0.3 # needs adjusting per device/scenario

final_result = ""

calc_result_running = False


fallback = False


def calc_result(counts):
    global calc_result_running, final_result
    
    calc_result_running = True
    
    filename = "test_" + str(counts) + ".wav"    
    result = pipe(filename)

    final_result = result["text"]
    
    calc_result_running = False
    
def audio_recording():
    global frames5, frames10, all_frames, all_speech_probs, recording, fallback
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    print("Recording started...")
    #all_speech_probs = []
    tot_len = 0
    while True:
        if not recording:
            continue
        data = stream.read(CHUNK)
        all_frames.append(data)
        #frames5.append(data) #frames5: 6-15, 16-25, 26-35, 36-45, ...
        frames10.append(data) #frames10: 1-10, 11-20, 21-30, 31-40, ...
        
        tot_len += 1
        if tot_len >= 6:
            frames5.append(data)
            
        if tot_len % 10 == 0 and not fallback:
            filename = "test_" + str(tot_len) + ".wav"
            wf = wave.open(filename, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(all_frames))
            wf.close()
            if calc_result_running and my_thread:
                if my_thread.is_alive():
                    fallback = True
                    print("Falling back to processing after finish.")
                #print("Killing thread: " + str(my_thread.ident))
                #ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(my_thread.ident), ctypes.py_object(SystemExit))
            if not fallback:
                my_thread = threading.Thread(target=calc_result, args=(tot_len,))
                #print("Running new thread")
                my_thread.start()
            
        # 200-250ms intervals fed into vad is usually good
        if len(frames5) == 10:
            frames5_data = b''.join(frames5)
            numbers = [int.from_bytes(frames5_data[i:i+2], byteorder='little', signed=False) for i in range(0, len(frames5_data), 2)]

            # Normalize integers to range between 0 and 2
            max_val = max(numbers)
            min_val = min(numbers)
            normalized_numbers = [(num - min_val) * 2 / (max_val - min_val) for num in numbers]

            wav = torch.tensor(normalized_numbers, dtype=torch.float32)
            
            window_size_samples = 1536
            speech_probs = []
            for i in range(0, len(wav), window_size_samples):
                chunk = wav[i: i+ window_size_samples]
                if len(chunk) < window_size_samples:
                    break
                speech_prob = model(chunk, 16000).item()
                speech_probs.append(speech_prob)
            vad_iterator.reset_states()
            all_speech_probs += speech_probs[:10]

            #print(max(speech_probs[:10]))
            
            if max(speech_probs[:10]) <= BOUNDARY:
                stop_recording()

            frames5 = []
            
        if len(frames10) == 10:
            frames10_data = b''.join(frames10)
            numbers = [int.from_bytes(frames10_data[i:i+2], byteorder='little', signed=False) for i in range(0, len(frames10_data), 2)]

            # Normalize integers to range between 0 and 2
            max_val = max(numbers)
            min_val = min(numbers)
            normalized_numbers = [(num - min_val) * 2 / (max_val - min_val) for num in numbers]

            wav = torch.tensor(normalized_numbers, dtype=torch.float32)
       
            window_size_samples = 1536
            speech_probs = []
            for i in range(0, len(wav), window_size_samples):
                chunk = wav[i: i+ window_size_samples]
                if len(chunk) < window_size_samples:
                    break
                speech_prob = model(chunk, 16000).item()
                speech_probs.append(speech_prob)
            vad_iterator.reset_states()
            all_speech_probs += speech_probs[:10]

            #print(max(speech_probs[:10]))
            
            if max(speech_probs[:10]) <= BOUNDARY:
                stop_recording()
            
            frames10 = []

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('start_recording')
def start_recording():
    global frames5, frames10, all_frames, recording
    frames5 = []  # Clear existing frames
    frames10 = []
    all_frames = []
    recording = True
    threading.Thread(target=audio_recording).start()

@socketio.on('stop_recording')
def stop_recording():
    global recording
    recording = False
    print("Recording stopped")
    
    if not fallback:
        while calc_result_running:
            continue
        print(final_result)
    
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

if __name__ == '__main__':
    socketio.run(app, debug=True)
