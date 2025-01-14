import torch
import torchaudio
from pathlib import Path
from resemble_enhance.enhancer.inference import enhance

device = "cuda" if torch.cuda.is_available() else "cpu"

def enhance_audio(input_audio_path, output_audio_path, solver="midpoint", nfe=64, tau=0.5):
    dwav, sr = torchaudio.load(input_audio_path)
    dwav = dwav.mean(dim=0)
    
    enhanced_audio, new_sr = enhance(dwav, sr, device, nfe=nfe, solver=solver, lambd=0.1, tau=tau)
    
    torchaudio.save(output_audio_path, enhanced_audio.unsqueeze(0), new_sr)
    print(f"Enhanced audio saved to: {output_audio_path}")
