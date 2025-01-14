import torch
import torchaudio
from pathlib import Path
from resemble_enhance.enhancer.inference import denoise

device = "cuda" if torch.cuda.is_available() else "cpu"

def denoise_audio(input_audio_path, output_audio_path):
    dwav, sr = torchaudio.load(input_audio_path)
    dwav = dwav.mean(dim=0)

    denoised_audio, new_sr = denoise(dwav, sr, device)
    
    torchaudio.save(output_audio_path, denoised_audio.unsqueeze(0), new_sr)
    print(f"Denoised audio saved to: {output_audio_path}")
