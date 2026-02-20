import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model

model_name = "facebook/wav2vec2-base"

processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2Model.from_pretrained(model_name)
model.eval()

def extract_embeddings(audio):
    wave, rate = torchaudio.load(audio)
    if rate != 16000:   
        wave = torchaudio.functional.resample(wave, rate, 16000)
        rate=16000

    wave = wave.mean(dim=0)   
    inputs = processor(wave, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)
    
    return embedding.squeeze(0)
