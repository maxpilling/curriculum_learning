"""get_model_info

A general utility script to get information from the TensorFlow model files.

"""
import sys
import os

import tensorflow as tf
import numpy as np

from PIL import Image

VERBOSE = False
LOG_DIR = "mass_log_dir"


def get_weights():
    """get_weights

    Get the weights from the current session.
    """

    return [
        v
        for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        if v.name.endswith("weights:0")
    ]


def get_conv_weights(weights, sess):
    """get_conv_weights

    Get the convolutional weights from the given weights.
    """

    conv_weights = [
        {"weights": v.eval(session=sess), "name": v.name}
        for v in weights
        if v.name.find("conv_layer")
    ]
    return [w for w in conv_weights if w["weights"].ndim == 4]


def get_filters_from_layer(layer_dict):
    """get_filters_from_layer

    Get the filters from the convolutional layers.
    """

    transposed_weights = layer_dict["weights"].transpose()

    image_list = []

    for image_row in transposed_weights:
        for sub_image in image_row:
            scaled_image = (
                255.0 / sub_image.max() * (sub_image - sub_image.min())
            ).astype(np.uint8)
            image_list.append(scaled_image)

    return image_list


def visualise_filter_from_layer(images, folder_name, filter_name):
    """visualise_filter_from_layer

    Visualises the gives filter as a grey scale image.
    """

    save_dir = os.path.join(LOG_DIR, folder_name, "images", filter_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, image in enumerate(images):
        file_name = f"{save_dir}/{i}.png"
        size = image.shape[0]

        img = Image.fromarray(image)
        img = img.resize((size * 10, size * 10))
        img.convert("L")

        img.save(file_name)


def print_all_info(graph, weights, sess):
    """print_all_info

    Get all the info from a TensorFlow model.
    """

    print("All operations: ")
    for operation in graph.get_operations():
        print(operation.name)

    print("All variables: ")
    for variable in tf.global_variables():
        print(variable.name)

    print("Trainable variables: ")
    for variable in tf.trainable_variables():
        print(variable.name)

    print("All weights: ")
    for weight in weights:
        print(weight.eval(session=sess))


def main():
    """main"""

    print("Parsing args...")
    passed_folder = sys.argv[1]

    meta_graph_path = ""
    for file in os.listdir(passed_folder):
        if file.endswith(".meta"):
            current_path = os.path.join(passed_folder, file)
            if meta_graph_path == "":
                meta_graph_path = current_path
            else:
                checkpoint_higher = file.split("-")[-1] > meta_graph_path.split("-")[-1]
                meta_graph_path = current_path if checkpoint_higher else meta_graph_path

    model_name = passed_folder.split("\\")[-1]
    model_folder = "\\".join(meta_graph_path.split("\\")[:-1])

    print("Creating session...")
    session = tf.Session()

    print("Loading session...")
    saver = tf.train.import_meta_graph(meta_graph_path)
    saver.restore(session, tf.train.latest_checkpoint(model_folder))

    print("Writing graph log...")
    writer = tf.summary.FileWriter(LOG_DIR, session.graph)
    writer.close()

    print("Getting all weights...")
    weights = get_weights()
    conv_layer_dicts = get_conv_weights(weights, session)
    graph = tf.get_default_graph()
    # all_variables = tf.all_variables()

    if VERBOSE:
        print("Dumping graph as txt...")
        out_file = "train.pbtxt"
        input_graph_def = graph.as_graph_def()
        tf.train.write_graph(
            input_graph_def, logdir=LOG_DIR, name=out_file, as_text=True
        )

        print("Printing everything...")
        print_all_info(graph, weights, session)

    print("Getting all images in Conv Filters...")

    for i, layer_dict in enumerate(conv_layer_dicts):
        print(f"    Processing layer {i + 1} out of {len(conv_layer_dicts)}...")
        images = get_filters_from_layer(layer_dict)
        layer_name = "_".join(layer_dict["name"].split("/")[-3:-1])
        visualise_filter_from_layer(images, model_name, layer_name)


if __name__ == "__main__":
    main()
