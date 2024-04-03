import torch
from pprint import pprint
import matplotlib.pyplot as plt

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

wav = read_audio('en.wav', sampling_rate=16000)
# get speech timestamps from full audio file
#speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=16000)
#pprint(speech_timestamps)

speech_probs = []
window_size_samples = 1536
for i in range(0, len(wav), window_size_samples):
    chunk = wav[i: i+ window_size_samples]
    if len(chunk) < window_size_samples:
      break
    speech_prob = model(chunk, 16000).item()
    speech_probs.append(speech_prob)
vad_iterator.reset_states() # reset model states after each audio

#print(speech_probs[:10]) # first 10 chunks predicts

plt.plot([i for i in range(len(speech_probs))], speech_probs)

plt.show()


