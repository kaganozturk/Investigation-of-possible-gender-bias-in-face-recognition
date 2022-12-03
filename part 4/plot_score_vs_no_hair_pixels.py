import pickle
from matplotlib import pyplot as plt
import numpy as np


def no_hair_pixels_difference(p1, p2, total_no_pixels=512*512):
    return round(abs((p1-p2)/total_no_pixels), 3)


with open('matching_scores.pkl', 'rb') as f:
    matching_scores = pickle.load(f)

with open('no_hair_pixels.pkl', 'rb') as f:
    no_hair_pixels = pickle.load(f)

hair_pixels_gen = []
hair_pixels_imp_same_hair_style = []
hair_pixels_imp_shaved = []
hair_pixels_imp = []
scores_gen = []
scores_imp_same_hair_style = []
scores_imp_shaved = []
scores_imp = []
for k, v in matching_scores.items():
    p1, p2 = k.split('vs')
    no_hair_p1 = no_hair_pixels[p1]
    no_hair_p2 = no_hair_pixels[p2]
    split_p1 = p1.split('_')
    split_p2 = p2.split('_')
    id1 = split_p1[0]
    id2 = split_p2[0]
    if id1 == id2:
        hair_pixels_gen.append(no_hair_pixels_difference(no_hair_p1, no_hair_p2))
        scores_gen.append(v)
    else:
        is_p1_has_hair = len(split_p1) == 2
        is_p2_has_hair = len(split_p2) == 2
        if is_p1_has_hair and is_p2_has_hair and split_p1[1]==split_p2[1]:
            hair_pixels_imp_same_hair_style.append(no_hair_pixels_difference(no_hair_p1, no_hair_p2))
            scores_imp_same_hair_style.append(v)
        elif not (is_p1_has_hair or is_p2_has_hair):
            hair_pixels_imp_shaved.append(no_hair_pixels_difference(no_hair_p1, no_hair_p2))
            scores_imp_shaved.append(v)
        else:
            hair_pixels_imp.append(no_hair_pixels_difference(no_hair_p1, no_hair_p2))
            scores_imp.append(v)

plt.scatter(hair_pixels_imp, scores_imp, c='b', marker='+', alpha=.3)
plt.scatter(hair_pixels_imp_same_hair_style, scores_imp_same_hair_style, c='g', marker='+', alpha=.3)
plt.scatter(hair_pixels_imp_shaved, scores_imp_shaved, c='b', marker='+', alpha=.3)
plt.scatter(hair_pixels_gen, scores_gen, c='r', marker='o', alpha=.5)
plt.xlabel('hair_pixel_differance')
plt.ylabel('matching_score')
plt.savefig('plot.png')
plt.show()
