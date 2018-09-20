import sys
import os
import glob
import json

import tensorflow as tf
import numpy as np

from PIL import Image, ImageDraw

VERBOSE = False
LOG_DIR = "mass_log_dir"


def get_weights():
    return [
        v
        for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        if v.name.endswith("weights:0")
    ]


def get_conv_weights(weights, sess):
    conv_weights = [
        {"weights": v.eval(session=sess), "name": v.name}
        for v in weights
        if v.name.find("conv_layer")
    ]
    return [w for w in conv_weights if w["weights"].ndim == 4]


def get_model_path(folder):
    meta_graph_path = ""

    for file in os.listdir(folder):
        if file.endswith(".meta"):
            current_path = os.path.join(folder, file)
            if meta_graph_path == "":
                meta_graph_path = current_path
            else:
                checkpoint_higher = file.split("-")[-1] > meta_graph_path.split("-")[-1]
                meta_graph_path = current_path if checkpoint_higher else meta_graph_path

    return meta_graph_path


def main():

    print("Parsing args...")
    base_model_folder = sys.argv[1]

    base_meta_graph_path = get_model_path(base_model_folder)

    base_model_folder = "\\".join(base_meta_graph_path.split("\\")[:-1])

    print("Creating session...")
    session_base = tf.Session()

    with session_base as sess:
        print("Loading graph...")
        saver_base = tf.train.import_meta_graph(base_meta_graph_path)

        print("Restoring graph...")
        saver_base.restore(sess, tf.train.latest_checkpoint(base_model_folder))

        print("Getting all weights...")
        with sess.graph.as_default():
            weights_base = get_weights()
            conv_weights_base = get_conv_weights(weights_base, sess)

    with open(f"{sys.argv[2]}.npy", 'wb') as base_file:
        np.save(base_file, conv_weights_base)


if __name__ == "__main__":
    main()
