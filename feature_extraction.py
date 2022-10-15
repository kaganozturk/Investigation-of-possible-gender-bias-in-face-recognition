"""
Uses weights and models implementation' from
https://github.com/deepinsight/insightface
"""

import argparse
from os import listdir, makedirs, path
import cv2
import numpy as np
import face_model


def extract_features(model, source, destination):
    source_list = listdir(source)
    source_list = [i for i in source_list if i.endswith('.jpg')]
    for image_name in source_list:
        image_path = path.join(source, image_name)
        features_name = path.join(destination, image_name[:-3] + "npy")
        img = cv2.imread(image_path)
        features = model.get_feature(img)
        np.save(features_name, features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features with CNN")
    parser.add_argument("--source", "-s", default="examples", help="Folder with images.")
    parser.add_argument("--dest", "-d", default="results", help="Folder to save the extractions.")

    # ArcFace params
    parser.add_argument(
        "--model",
        help="path to model.",
        default="models/model-r100-ii/model,0",
    )

    args = parser.parse_args()
    model = face_model.FaceModel(args)

    if not path.exists(args.dest):
        makedirs(args.dest)

    extract_features(model, args.source, args.dest)
