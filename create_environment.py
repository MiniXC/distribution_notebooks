from audiomentations import RoomSimulator, AddGaussianSNR, Compose
import torchaudio
import torchaudio.functional as F
import torch
from tqdm.auto import tqdm
import numpy as np

import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--synth_dataset", type=str, default=None)
parser.add_argument("--target_dataset", type=str, default=None)
parser.add_argument("--fs", type=int, default=16_000)

args = parser.parse_args()

args = parser.parse_args()

speaker_augmentations = {}

def augment_wav(wav, speaker):
    first_augment = False
    if speaker not in speaker_augmentations:
        speaker_augmentations[speaker] = Compose([
            AddGaussianSNR(p=1),
            RoomSimulator(max_target_rt60=0.8, calculation_mode="rt60", p=0.8),
        ])
        # speaker_augmentations[speaker] = RoomSimulator(max_target_rt60=1, calculation_mode="rt60", p=0.8)
        first_augment = True
    augment = speaker_augmentations[speaker]
    wav, fs = torchaudio.load(wav)
    wav = F.resample(wav, fs, args.fs)
    wav_sum = np.sum(np.abs(wav.numpy()))
    wav = augment(samples=wav.numpy()[0], sample_rate=args.fs)
    # scale to original mean
    wav = wav * (wav_sum / np.abs(wav).sum())
    if first_augment:
        speaker_augmentations[speaker].freeze_parameters()
    return wav

for synth_wav in tqdm(list(Path(args.synth_dataset).rglob("*.wav"))):
    if "original" in synth_wav.name:
        continue

    speaker = synth_wav.parent.name
    synth_tg = synth_wav.with_suffix(".meta")
    synth_lab = synth_wav.with_suffix(".lab")
    wav = augment_wav(synth_wav, speaker)

    target_wav = Path(args.target_dataset) / speaker / synth_wav.name
    target_wav.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(target_wav, torch.tensor(wav).unsqueeze(0), args.fs)

    target_tg = target_wav.with_suffix(".meta")
    target_lab = target_wav.with_suffix(".lab")
    if synth_tg.exists():
        target_tg.symlink_to(synth_tg.resolve())
    if synth_lab.exists():
        target_lab.symlink_to(synth_lab.resolve())