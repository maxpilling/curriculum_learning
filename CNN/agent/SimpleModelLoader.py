import os

import tensorflow as tf


class SimpleModelLoader():
    """A simple model loader for TF."""

    def __init__(self, model_meta_path, graph_to_use):
        self.graph = graph_to_use
        self.session = tf.Session(graph=self.graph)

        # TODO: Pass over and have theta depend on model number.
        self.previous_model_number = None

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

    @property
    def flatten_1(self):
        return self.graph.get_tensor_by_name('theta/Flatten_1/flatten/Reshape:0')

    @property
    def concat_2(self):
        return self.graph.get_tensor_by_name('theta/concat_2:0')
