import numpy as np
import argparse
from pathlib import Path
import torchaudio
import torchaudio.functional as F
from tqdm.contrib.concurrent import process_map
from tqdm.auto import tqdm
import torch
import pickle

wav2mel = torch.jit.load("models/wav2mel.pt").eval()
dvector = torch.jit.load("models/dvector.pt").eval()

from measures import measures

parser = argparse.ArgumentParser()
parser.add_argument("--synth_dataset", type=str, default=None)
parser.add_argument("--real_dataset", type=str, default=None)
parser.add_argument("--fs", type=int, default=16_000)
parser.add_argument("--name", type=str, default="default")

args = parser.parse_args()

def compute_measure(synth_wav):
    try:
        # real values
        real_wav = Path(args.real_dataset) / synth_wav.relative_to(args.synth_dataset)
        speaker = real_wav.parent.name
        real_tg = real_wav.with_suffix(".TextGrid")
        real_dvec = real_wav.with_suffix(".npy")
        
        # synth values
        synth_tg = synth_wav.with_suffix(".meta")
        synth_dvec = synth_wav.with_suffix(".npy")

        # load audio
        real_wav, real_fs = torchaudio.load(real_wav)
        synth_wav, synth_fs = torchaudio.load(synth_wav)
        real_wav = F.resample(real_wav, real_fs, args.fs)
        synth_wav = F.resample(synth_wav, synth_fs, args.fs)

        # compute measures
        result = {
            "real": {},
            "synth": {},
        }
        for measure, func in measures.items():
            if measure == "dvec":
                if real_dvec.exists():
                    real = np.load(real_dvec)
                else:
                    real = dvector.embed_utterance(wav2mel(real_wav, args.fs)).detach().numpy()
                if synth_dvec.exists():
                    synth = np.load(synth_dvec)
                else:
                    synth = dvector.embed_utterance(wav2mel(synth_wav, args.fs)).detach().numpy()
            elif measure == "duration":
                real = real_tg
                synth = synth_tg
            else:
                real = real_wav.numpy()[0]
                synth = synth_wav.numpy()[0]
            result["real"][measure] = func(real)
            result["synth"][measure] = func(synth)
        
        result["speaker"] = speaker

        return result
    except RuntimeError:
        return None
       

all_wavs  = []

for synth_wav in Path(args.synth_dataset).rglob("*.wav"):
    if "original" in synth_wav.name:
        continue
    all_wavs.append(synth_wav)

# for wav in tqdm(all_wavs):
#     result = compute_measure(wav)
#     # print(result)

results = process_map(compute_measure, all_wavs, chunksize=1, max_workers=2)

with open(f"{args.name}.pkl", "wb") as f:
    pickle.dump(results, f)