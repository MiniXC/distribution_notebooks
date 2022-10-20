from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
import numpy as np
import pickle
import argparse

def normalize(x, y):
    mu, sigma = np.mean(x), np.std(x)
    x = (x - mu) / sigma
    y = (y - mu) / sigma
    return x, y

def w1(x, y):
    x, y = normalize(x, y)
    return wasserstein_distance(x, y)

def l1(x, y):
    x, y = normalize(x, y)
    return np.linalg.norm(x - y, ord=1) / len(x)

parser = argparse.ArgumentParser()
parser.add_argument("--results", type=str, default=None)

args = parser.parse_args()

with open(args.results, "rb") as f:
    results = pickle.load(f)

for measure in results[0]["real"].keys():
    real = np.array([r["real"][measure] for r in results if r is not None])
    synth = np.array([r["synth"][measure] for r in results if r is not None])
    if measure == "dvec":
        # pca
        pca = PCA(n_components=2)
        real = pca.fit_transform(real)
        synth = pca.transform(synth)
        # dict
        speakers = [r["speaker"] for r in results if r is not None]
        speaker_dict_real = {s: [] for s in set(speakers)}
        speaker_dict_synth = {s: [] for s in set(speakers)}
        for i, speaker in enumerate(speakers):
            speaker_dict_real[speaker].append(real[i])
            speaker_dict_synth[speaker].append(synth[i])
        # intra-speaker
        speaker_means_real = {}
        speaker_means_synth = {}
        for speaker, vectors in speaker_dict_real.items():
            speaker_means_real[speaker] = np.mean(vectors, axis=0)
        for speaker, vectors in speaker_dict_synth.items():
            speaker_dict_synth[speaker] = np.mean(vectors, axis=0)
        real_intra = np.array([r - speaker_means_real[s] for s, r in zip(speakers, real)])
        synth_intra = np.array([r - speaker_dict_synth[s] for s, r in zip(speakers, synth)])
        #print(f"{measure} intra-speaker (pc1) w1: {w1(real_intra[:, 0], synth_intra[:, 0]):.3f}")
        #print(f"{measure} intra-speaker (pc2) w1: {w1(real_intra[:, 1], synth_intra[:, 1]):.3f}")
        print(f"{measure} intra-speaker (pc1) l1: {l1(real_intra[:, 0], synth_intra[:, 0]):.3f}")
        print(f"{measure} intra-speaker (pc2) l1: {l1(real_intra[:, 1], synth_intra[:, 1]):.3f}")
        # inter-speaker
        real_inter = np.array([speaker_means_real[s] for s in set(speakers)])
        synth_inter = np.array([speaker_dict_synth[s] for s in set(speakers)])
        #print(f"{measure} inter-speaker (pc1) w1: {w1(real_inter[:, 0], synth_inter[:, 0]):.3f}")
        #print(f"{measure} inter-speaker (pc2) w1: {w1(real_inter[:, 1], synth_inter[:, 1]):.3f}")
        print(f"{measure} inter-speaker (pc1) l1: {l1(real_inter[:, 0], synth_inter[:, 0]):.3f}")
        print(f"{measure} inter-speaker (pc2) l1: {l1(real_inter[:, 1], synth_inter[:, 1]):.3f}")
    else:
        #print(f"{measure} w1: {w1(real, synth):.3f}")
        print(f"{measure} l1: {l1(real, synth):.3f}")