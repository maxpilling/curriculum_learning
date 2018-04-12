import sys
import os

import tensorflow as tf
import numpy as np

from PIL import Image, ImageDraw

VERBOSE = False
LOG_DIR = "mass_log_dir"

def get_weights():
    return [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.endswith('weights:0')]

def get_conv_weights(weights, sess):
    conv_weights = [v.eval(session=sess) for v in weights if v.name.find("conv_layer")]
    return [w for w in conv_weights if w.ndim == 4]

def get_filters_from_layer(weights):
    transposed_weights = weights.transpose()

    image_list = []

    for image_row in transposed_weights:
        for sub_image in image_row:
            image_list.append(sub_image)

    return image_list

def visualise_filter_from_layer(images, index):

    save_dir = os.path.join(LOG_DIR, "images", str(index))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, image in enumerate(images):
        file_name = f"{save_dir}/{i}.png"
        size = image.shape[0]

        img = Image.fromarray(image, 'RGB')
        img = img.resize((size * 10, size * 10))
        img.save(file_name)

def print_all_info(graph, weights, sess):
    print("All operations: ")
    for op in graph.get_operations():
        print(op.name)

    print("All variables: ")
    for v in tf.global_variables():
        print(v.name)

    print("Trainable variables: ")
    for v in tf.trainable_variables():
        print(v.name)

    print("All weights: ")
    for weight in weights:
        print(weight.eval(session=sess))

def main():

    print("Parsing args...")
    meta_graph = sys.argv[1]
    model_folder = '\\'.join(meta_graph.split('\\')[:-1])

    print("Creating session...")
    session = tf.Session()

    print("Loading session...")
    saver = tf.train.import_meta_graph(meta_graph)
    saver.restore(session, tf.train.latest_checkpoint(model_folder))

    print("Writing graph log...")
    writer = tf.summary.FileWriter(LOG_DIR, session.graph)
    writer.close()

    print("Getting all weights...")
    weights = get_weights()
    conv_weights = get_conv_weights(weights, session)
    graph = tf.get_default_graph()
    # all_variables = tf.all_variables()

    if (VERBOSE):
        print("Dumping graph as txt...")
        out_file = "train.pbtxt"
        input_graph_def = graph.as_graph_def()
        tf.train.write_graph(input_graph_def, logdir=LOG_DIR, name=out_file, as_text=True)

        print("Printing everything...")
        print_all_info(graph, weights, session)

    print("Getting all images in Conv Filters...")

    for i, layer_weights in enumerate(conv_weights):
        print(f"    Processing layer {i + 1} out of {len(conv_weights)}...")
        images = get_filters_from_layer(layer_weights)
        visualise_filter_from_layer(images, i)


if __name__ == "__main__":
    main()
