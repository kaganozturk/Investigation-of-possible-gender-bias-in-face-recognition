import os
import itertools
import pickle
import numpy as np
import cv2
from minplus_utils import read_img, imshow, heatmap, contours, color_mask
from minplus_utils import saliency_minus, saliency_plus, saliency_RISE
from arcface import ArcFace
import argparse

arcface = ArcFace.ArcFace()


# 2) MATCHING SCORE FUNCTION FOR FACE VERIFICATION
def fx_face_matching(X):
    # y is the reference embedding
    x = arcface.calc_emb(X)
    score = np.dot(x, y)
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate matching scores")
    parser.add_argument("--vis", "-v", default=0, help="Create Heatmaps", action='store_true')

    args = parser.parse_args()
    fpath_in = 'data_example/'
    no_person = 2
    no_hair_style = 3
    data_lst = sorted(os.listdir(fpath_in))
    scores = {}
    for p_0, i in enumerate(data_lst):
        img_Y = fpath_in + i
        for p_1, k in enumerate(data_lst[p_0 + 1:]):
            img_X = fpath_in + k
            pos = (15, 25)
            Y = read_img(img_Y)
            X = read_img(img_X)
            # (N, M) = Y.shape[0:2]
            Ys = Y.copy()
            Xs = X.copy()
            cv2.putText(Ys, 'Y', pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(Xs, 'X', pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            Xin = cv2.hconcat([Ys, Xs])

            y = arcface.calc_emb(Y)
            score = fx_face_matching(X)
            scores['{}vs{}'.format(os.path.splitext(i)[0], os.path.splitext(k)[0])] = round(score, 3)
            print('{}vs{} matching_score = {:.2f}'.format(os.path.splitext(i)[0],
                                                          os.path.splitext(k)[0],
                                                          score))

            if args.vis:
                # 4) PARAMETERS OF MINPLUS
                FAST_MODE = False

                if FAST_MODE:  # 1 minute
                    gsigma = 221  # width of Gaussian mask
                    d = 48  # steps (one evaluation each d x d pixeles)
                    tmax = 1  # maximal number of iterations
                else:  # 6 minutes
                    gsigma = 161  # width of Gaussian mask
                    d = 16  # steps (one evaluation each d x d pixeles)
                    tmax = 20  # maximal number of iterations
                dsc = 0.01
                fx = fx_face_matching

                # Minus
                H0m, H1m = saliency_minus(X, fx, nh=gsigma, d=d, n=tmax, nmod=2, th=dsc)
                S0m, Y0m = heatmap(X, H0m, gsigma)
                S1m, Y1m = heatmap(X, H1m, gsigma)
                pos = (15, 25)
                cv2.putText(Y0m, 'S0-', pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(Y1m, 'S1-', pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                Im = cv2.hconcat([X, Y0m, Y1m])
                # imshow(Im,fpath='heatmap_minus.png',show_pause=1)

                # Plus
                H0p, H1p = saliency_plus(X, fx, nh=gsigma, d=d, n=tmax, nmod=1, th=dsc)
                S0p, Y0p = heatmap(X, H0p, gsigma)
                S1p, Y1p = heatmap(X, H1p, gsigma)
                cv2.putText(Y0p, 'S0+', pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(Y1p, 'S1+', pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                Ip = cv2.hconcat([Xin, Y0p, Y1p])
                # imshow(Ip,fpath='heatmap_plus.png',show_pause=1)

                # MinPlus
                # ------------------------------------------------------
                # WARNING: Run saliency-minus and saliency-plus before
                # ------------------------------------------------------

                Havg = (H0m + H0p + H1m + H1p) / 4  # <= HeatMap between 0 and 1
                Smp, Ymp = heatmap(X, Havg, gsigma)
                cv2.putText(Ymp, 'MinPlus', pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                C = contours(255 * Smp, X, 'jet', 10, print_levels=False, color_levels=True)
                Imp = cv2.hconcat([Xin, Ymp, C])
                imshow(Imp, title='matching_score = {:.2f}'.format(score), fpath=None)

    with open('matching_scores.pkl', 'wb') as f:
        pickle.dump(scores, f)


