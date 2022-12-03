import argparse
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def match_features(args):
    im1_feat = np.load(args.im1)
    im2_feat = np.load(args.im2)
    return cosine_similarity(im1_feat, im2_feat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cosine similarity of two features")
    parser.add_argument("--im1", "-r", default="results/test_img_1.npy")
    parser.add_argument("--im2", "-q", default="results/test_img_2.npy")

    args = parser.parse_args()
    score = match_features(args)
    print(score)
