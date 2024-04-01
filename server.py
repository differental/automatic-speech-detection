from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import pyaudio
import wave
import threading
# ---
import torch
import matplotlib.pyplot as plt

import whisper

model_whisper = whisper.load_model("tiny")

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
# ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

#1 frame: 1024/44100=23ms
#10 frames together, send in 5 frame intervals (1-10, 6-15, 11-20...)

# Comments:
# In production, transcription (whisper) >= 115ms


# Go ahead with:
# Distilled whisper medium

# Sheet: delay / inaccuracy

# Do transcription as the same time as person speaking

# Overflow issues: time-stuff



# Next problem: more real-world 




frames5 = []
frames10 = []
all_frames = []
all_speech_probs = []
recording = False

def audio_recording():
    global frames5, frames10, all_frames, all_speech_probs, recording
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    print("Recording started...")
    #all_speech_probs = []
    tot_len = 0
    while True:
        data = stream.read(CHUNK)
        all_frames.append(data)
        #frames5.append(data) #frames5: 6-15, 16-25, 26-35, 36-45, ...
        frames10.append(data) #frames10: 1-10, 11-20, 21-30, 31-40, ...
        
        if not recording:
            continue

        tot_len += 1
        if tot_len >= 6:
            frames5.append(data)

        if len(frames5) == 10:
            # measure latency of this
            wf = wave.open("test5.wav", 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames5))
            wf.close()
            #print('audio saved 5')
            frames5 = []

            # it's possible to plug numpy arrays to make the two match. figure
            # out this (if 20-30ms)
            # A100 might be slow
            wav = read_audio('test5.wav', sampling_rate=16000)
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
            #print(speech_probs)

            flag=True
            for i in speech_probs[:10]:
                if i > 0.01:
                    flag=False
            if flag: # stop recording 
                stop_recording()

 
        if len(frames10) == 10:
            wf = wave.open("test10.wav", 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames10))
            wf.close()
            #print('audio saved 10')
            frames10 = []

            wav = read_audio('test10.wav', sampling_rate=16000)
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
            #print(speech_probs)

            flag=True
            for i in speech_probs[:10]:
                if i > 0.01:
                    flag=False
            if flag: stop_recording()

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
    global frames5, frames10, all_frames, all_speech_probs, recording
    print("Recording stopped...")
    filename = 'recording_all.wav'
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(all_frames))
    wf.close()
    
    result = model_whisper.transcribe("recording_all.wav")
    print(result['text'])
    recording = False
    #emit('audio_saved', {'filename': filename})
    #plt.plot([i for i in range(len(all_speech_probs))], all_speech_probs)
    #plt.show()


if __name__ == '__main__':
    socketio.run(app, debug=True)

