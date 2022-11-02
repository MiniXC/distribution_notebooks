from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
import numpy as np
import pickle
import argparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import linalg

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(vecs):
    mu = np.mean(vecs, axis=0)
    sigma = np.cov(vecs, rowvar=False)
    return mu, sigma

def normalize(x, y):
    mu, sigma = np.mean(x), np.std(x)
    x = (x - mu) / sigma
    y = (y - mu) / sigma
    return x, y

def w1(x, y, p=2):
    x, y = normalize(x, y)
    #return wasserstein_distance(x, y)
    return np.sum(np.abs((np.sort(x)-np.sort(y))**p))/x.shape[0]

def l1(x, y):
    x, y = normalize(x, y)
    return np.linalg.norm(x - y, ord=1) / len(x)

parser = argparse.ArgumentParser()
parser.add_argument("--results", type=str, default=None)
parser.add_argument("--plot", action="store_true")
parser.add_argument("--distance_measure", type=str, default="w1")

args = parser.parse_args()

with open(args.results, "rb") as f:
    results = pickle.load(f)

def create_histplot(real, synth, ax, is_wada=False):
    real_vals, synth_vals = real, synth
    if not is_wada:
        min_val = real_vals.mean() - real_vals.std() * 3
        max_val = real_vals.mean() + real_vals.std() * 3
        min_val_s = synth_vals.mean() - synth_vals.std() * 3
        max_val_s = synth_vals.mean() + synth_vals.std() * 3
        real_vals_new = real_vals[((real_vals>min_val)&(real_vals<max_val))&((synth_vals>min_val_s)&(synth_vals<max_val_s))]
        synth_vals_new = synth_vals[((real_vals>min_val)&(real_vals<max_val))&((synth_vals>min_val_s)&(synth_vals<max_val_s))]
    else:
        real_vals_new = real_vals
        synth_vals_new = synth_vals
    return sns.histplot(
        x=np.concatenate((real_vals_new,synth_vals_new)),
        hue=["Real"]*real_vals_new.shape[0]+["Synthetic"]*synth_vals_new.shape[0],
        ax=ax,
        stat="density",
        legend=None,
        linewidth=0,
    )

if args.plot:
    measures = len(results[0]["real"].keys())+1
    fig = plt.figure(constrained_layout=True, figsize=(15,2.5))
    gs = fig.add_gridspec(1, measures)

speakers_done = False

for j, measure in enumerate(results[0]["real"].keys()):
    real = np.array([r["real"][measure] for r in results if r is not None])
    synth = np.array([r["synth"][measure] for r in results if r is not None])
    if measure == "dvec":
        speakers_done = True
        #real, synth = normalize(real, synth)
        m1, s1 = calculate_activation_statistics(real)
        m2, s2 = calculate_activation_statistics(synth)
        #print(f"FID (overall): {calculate_frechet_distance(m1, s1, m2, s2):.2f}")
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
        real_intra, synth_intra = normalize(real_intra, synth_intra)
        m1, s1 = calculate_activation_statistics(real_intra)
        m2, s2 = calculate_activation_statistics(synth_intra)
        print(f"FID (intra): {calculate_frechet_distance(m1, s1, m2, s2):.2f}")
        if args.plot:
            ax1 = fig.add_subplot(gs[0, j])
            ax1.set_title("Intra-Speaker", y=0.8)
            sns.kdeplot(
                x=np.concatenate((real_intra[:,0], synth_intra[:,0])),
                y=np.concatenate((real_intra[:,1], synth_intra[:,1])),
                hue=["Real"]*real_intra.shape[0]+["Synthetic"]*synth_intra.shape[0],
                ax=ax1,
                legend=None,
                fill=True,
                thresh=0.005,
                alpha=0.8
            )
            ax1.set_xlabel("PC1")
            ax1.set_ylabel("PC2")
        # if args.distance_measure == "w1":
        #     print(f"{measure} intra-speaker (pc1) w1: {w1(real_intra[:, 0], synth_intra[:, 0]):.2f}")
        #     print(f"{measure} intra-speaker (pc2) w1: {w1(real_intra[:, 1], synth_intra[:, 1]):.2f}")
        # elif args.distance_measure == "l1":
        #     print(f"{measure} intra-speaker (pc1) l1: {l1(real_intra[:, 0], synth_intra[:, 0]):.2f}")
        #     print(f"{measure} intra-speaker (pc2) l1: {l1(real_intra[:, 1], synth_intra[:, 1]):.2f}")
        # inter-speaker
        real_inter = np.array([speaker_means_real[s] for s in set(speakers)])
        synth_inter = np.array([speaker_dict_synth[s] for s in set(speakers)])
        # normalize
        real_inter, synth_inter = normalize(real_inter, synth_inter)
        m1, s1 = calculate_activation_statistics(real_inter)
        m2, s2 = calculate_activation_statistics(synth_inter)
        print(f"FID (inter): {calculate_frechet_distance(m1, s1, m2, s2):.2f}")
        if args.plot:
            ax2 = fig.add_subplot(gs[0, j+1])
            ax2.set_title("Inter-Speaker", y=0.8)
            sns.kdeplot(
                x=np.concatenate((real_inter[:,0], synth_inter[:,0])),
                y=np.concatenate((real_inter[:,1], synth_inter[:,1])),
                hue=["Real"]*real_inter.shape[0]+["Synthetic"]*synth_inter.shape[0],
                ax=ax2,
                legend=None,
                fill=True,
                thresh=0.005,
                alpha=0.8
            )
            ax2.set_xlabel("PC1")
            ax2.set_ylabel("PC2")
        # if args.distance_measure == "w1":
        #     print(f"{measure} inter-speaker (pc1) w1: {w1(real_inter[:, 0], synth_inter[:, 0]):.2f}")
        #     print(f"{measure} inter-speaker (pc2) w1: {w1(real_inter[:, 1], synth_inter[:, 1]):.2f}")
        # elif args.distance_measure == "l1":
        #     print(f"{measure} inter-speaker (pc1) l1: {l1(real_inter[:, 0], synth_inter[:, 0]):.2f}")
        #     print(f"{measure} inter-speaker (pc2) l1: {l1(real_inter[:, 1], synth_inter[:, 1]):.2f}")
    else:
        if measure == "wada":
            min_val = -20+1e-6
            max_val = 100-1e-6
            min_val_s = -20+1e-6
            max_val_s = 100-1e-6
            real_vals_new = real[((real>min_val)&(real<max_val))&((synth>min_val_s)&(synth<max_val_s))]
            synth_vals_new = synth[((real>min_val)&(real<max_val))&((synth>min_val_s)&(synth<max_val_s))]
            real = real_vals_new
            synth = synth_vals_new
        if args.plot:
            if speakers_done:
                ax = fig.add_subplot(gs[0, j+1])
            else:
                ax = fig.add_subplot(gs[0, j])
            create_histplot(real, synth, ax, is_wada=measure=="wada")
            ax.set_xlabel(measure.upper())
        if args.distance_measure == "w1":
            print(f"{measure} w1: {w1(real, synth):.2f}")
        elif args.distance_measure == "l1":
            print(f"{measure} l1: {l1(real, synth):.2f}")

if args.plot:
    plt.show()