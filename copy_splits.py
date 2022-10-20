import pickle
from pathlib import Path
from tqdm.auto import tqdm

split_a = pickle.load(open("wavs_a.pkl", "rb"))
split_b = pickle.load(open("wavs_b.pkl", "rb"))
source_paths = [Path("../data/train-clean-100-aligned"), Path("../data/train-clean-360-aligned")]
data_path = Path("../data")

for split, wavs in [("train-clean-a", split_a), ("train-clean-b", split_b)]:
    for wav in tqdm(wavs, desc=split):
        wav = Path(wav)
        speaker = wav.parent.name
        name = wav.stem
        source_wav = None
        for source_path in source_paths:
            source_wav = (source_path / speaker / name).with_suffix(".wav")
            if source_wav.exists():
                break
        if not source_wav.exists():
            print(f"Could not find {source_wav}")
            continue
        target_path = (data_path / split / speaker / name)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        for ext in [".wav", ".npy", ".TextGrid", ".lab"]:
            source_file = source_wav.with_suffix(ext)
            target_file = target_path.with_suffix(ext)
            if source_file.exists():
                # symlink
                if not target_file.exists():
                    source_file = source_file.resolve()
                    target_file.symlink_to(source_file)
            else:
                print(f"Could not find {source_file}")