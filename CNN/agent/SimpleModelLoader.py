import os
import re

import tensorflow as tf


class SimpleModelLoader():
    """A simple model loader for TF."""

    def __init__(self, model_meta_path, graph_to_use, new_model_name):
        self.graph = graph_to_use
        self.session = tf.Session(graph=self.graph)
        self.new_model_name = new_model_name

        model_folder = os.path.dirname(model_meta_path)

        with self.graph.as_default():

            print("Importing Meta Graph...")
            saver = tf.train.import_meta_graph(
                model_meta_path
            )

            print("Restoring graph...")
            saver.restore(
                self.session,
                tf.train.latest_checkpoint(model_folder)
            )

    def get_all_tensors_by_name(self, tensor_name_regex):
        current_tensors = [n.name for n in self.graph.as_graph_def().node]
        tensors = []

        for tensor_name in current_tensors:
            if (re.findall(tensor_name_regex, tensor_name) and
                    not tensor_name.startswith(self.new_model_name)):

                tensors.append(
                    self.graph.get_tensor_by_name(f"{tensor_name}:0")
                )

        return tensors

    @property
    def flatten_1(self):
        return self.get_all_tensors_by_name(r"Flatten_1\/flatten\/Reshape$")

    @property
    def concat_2(self):
        return self.get_all_tensors_by_name(r"concat_2$")

    @property
    def screen_conv_1(self):
        return self.get_all_tensors_by_name(r"screen_network\/conv_layer1\/model_[0-9]+\/Relu")

    @property
    def minimap_conv_1(self):
        return self.get_all_tensors_by_name(r"minimap_network\/conv_layer1\/model_[0-9]+\/Relu")

    @property
    def fully_connected_layer1(self):
        return self.get_all_tensors_by_name(r"fully_connected_layer1\/model_[0-9]+\/Relu")
