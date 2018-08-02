import tensorflow as tf
from pysc2.lib import actions
from pysc2.lib.features import SCREEN_FEATURES, MINIMAP_FEATURES
from tensorflow.contrib import layers


class ConvPolicy:
    """ConvPolicy

    Tidied and altered code from https://github.com/pekaalto/sc2aibot

    The network structure of the agent, using convolutional
    layers against both the map and screen.
    """

    def __init__(self,
                 agent,
                 trainable: bool = True,
                 curriculum_number = -1
                 ):
        self.placeholders = agent.placeholders
        self.trainable = trainable
        self.unittype_emb_dim = agent.unit_type_emb_dim
        self.curriculum_number = curriculum_number

    @staticmethod
    def logclip(input_tensor):
        """logclip

        Given a tensor, restrict it to a given range, before
        taking the natural logarithm of each element.

        :param input_tensor: The tensor to be clipped and reduced.
        """

        return tf.log(tf.clip_by_value(input_tensor, 1e-12, 1.0))

    def build_conv_layers_for_input(self, inputs, name, previous_tensors):
        """build_conv_layers_for_input

        Creates 2 convolutional layers based on an input.
        Changeable parts here are:
            Number of outputs for both layers
            Size of the kernel used
            The stride used
            The activation function

        :param inputs: The inputs to run the convolutional layers against.
        :param name: The name of the input, to scope the layers.
        """

        conv_layer1 = layers.conv2d(
            inputs=inputs,
            data_format="NHWC",
            num_outputs=16,
            kernel_size=5,
            stride=1,
            padding="SAME",
            activation_fn=tf.nn.relu,
            scope=f"{name}/conv_layer1/model_{self.curriculum_number}",
            trainable=self.trainable
        )

        conv_layer2 = layers.conv2d(
            inputs=conv_layer1,
            data_format="NHWC",
            num_outputs=32,
            kernel_size=3,
            stride=1,
            padding="SAME",
            activation_fn=None,
            scope=f"{name}/conv_layer2/model_{self.curriculum_number}",
            trainable=self.trainable
        )

        # Sort the previous models
        previous_conv_layer2 = []
        for model_number, prev_out in enumerate(previous_tensors):
            conv_layer2_previous = layers.conv2d(
                inputs=prev_out,
                data_format="NHWC",
                num_outputs=32,
                kernel_size=3,
                stride=1,
                padding="SAME",
                activation_fn=None,
                scope=f"{name}/conv_layer2/model_{model_number}",
                trainable=self.trainable
            )

            previous_conv_layer2.append(conv_layer2_previous)

        previous_conv_layer2_added = self.add_all_previous(previous_conv_layer2)

        combined_conv_layer2 = tf.add(
            conv_layer2,
            previous_conv_layer2_added,
            '%s_conv_add' % name
        )

        relu_conv_layer2 = tf.nn.relu(
            combined_conv_layer2,
            name='combined_%s_conv_layer2_relu' % name
        )

        if self.trainable:
            layers.summarize_activation(conv_layer1)
            layers.summarize_activation(conv_layer2)
            layers.summarize_activation(relu_conv_layer2)
            tf.summary.image(f"{name}/new_conv_layer1", tf.reshape(conv_layer1, [-1, 32, 32, 1]), 3)
            tf.summary.image(f"{name}/new_conv_layer2", tf.reshape(conv_layer2, [-1, 32, 32, 1]), 3)
            tf.summary.image(f"{name}/combined_conv_layer2", tf.reshape(relu_conv_layer2, [-1, 32, 32, 1]), 3)

        return relu_conv_layer2

    def add_all_previous(self, previous_list):
        """add_all_previous

        For a list of tensors, add them all together, and return a final tensor.
        """

        if len(previous_list) == 1:
            return previous_list[0]

        final_tensor = tf.add(
            previous_list[0],
            previous_list[1]
        )

        for current_tensor in previous_list[2:]:
            final_tensor = tf.add(
                final_tensor,
                current_tensor
            )

        return final_tensor

    def build(self, session, previous_model):
        """build

        Build the actual network, using the
        values passed over the from agent object, which
        themselves are derived from the Obs object.
        """

        # Maps a series of symbols to embeddings,
        # where an embedding is a mapping from discrete objects,
        # such as words, to vectors of real numbers.
        # In this case it is from the unit types.
        units_embedded = layers.embed_sequence(
            self.placeholders.screen_unit_type,
            vocab_size=SCREEN_FEATURES.unit_type.scale,
            embed_dim=self.unittype_emb_dim,
            scope="unit_type_emb",
            trainable=self.trainable
        )

        # "One hot" encoding performs "binarization" on the input
        # meaning we end up with features we can suitably learn
        # from.
        # Basically, learning from categories isn't possible,
        # but learning from ints (i.e. 0/1/2 for 3 categories)
        # ends up with further issues, like the ML algorithm
        # picking up some pattern in the categories, when none exists.
        # Instead we want it in a binary form instead, to prevent this.
        # This is not needed for the background, since it is
        # not used, which is why we ignore channel 0 in the
        # last sub-array.
        player_relative_screen_one_hot = layers.one_hot_encoding(
            self.placeholders.player_relative_screen,
            num_classes=SCREEN_FEATURES.player_relative.scale
        )[:, :, :, 1:]

        player_relative_minimap_one_hot = layers.one_hot_encoding(
            self.placeholders.player_relative_minimap,
            num_classes=MINIMAP_FEATURES.player_relative.scale
        )[:, :, :, 1:]

        channel_axis = 3

        # Group together all the inputs, such that a conv
        # layer can be built upon them.
        screen_numeric_all = tf.concat(
            [
                self.placeholders.screen_numeric,
                units_embedded,
                player_relative_screen_one_hot
            ],
            axis=channel_axis
        )

        minimap_numeric_all = tf.concat(
            [
                self.placeholders.minimap_numeric,
                player_relative_minimap_one_hot
            ],
            axis=channel_axis
        )

        # Build the 2 convolutional layers based on the screen
        # and the mini-map.
        screen_conv_layer_output = self.build_conv_layers_for_input(
            screen_numeric_all,
            "screen_network",
            previous_model.screen_conv_1
        )

        # And now the minimap
        minimap_conv_layer_output = self.build_conv_layers_for_input(
            minimap_numeric_all,
            "minimap_network",
            previous_model.minimap_conv_1
        )

        # Group these two convolutional layers now, and
        # build a further convolutional layer on top of it.
        visual_inputs = tf.concat(
            [screen_conv_layer_output, minimap_conv_layer_output],
            axis=channel_axis
        )

        spatial_actions_normal = layers.conv2d(
            visual_inputs,
            data_format="NHWC",
            num_outputs=1,
            kernel_size=1,
            stride=1,
            activation_fn=None,
            scope=f"spatial_actions/model_{self.curriculum_number}",
            trainable=self.trainable
        )

        # Sort the previous models spatial action layers.
        previous_spatial_actions = []
        for model_number, prev_out in enumerate(previous_model.concat_2):
            spatial_actions_previous = layers.conv2d(
                prev_out,
                data_format="NHWC",
                num_outputs=1,
                kernel_size=1,
                stride=1,
                activation_fn=None,
                scope=f"spatial_actions/model_{model_number}",
                trainable=self.trainable
            )

            previous_spatial_actions.append(spatial_actions_previous)

        previous_spatial_actions_added = self.add_all_previous(previous_spatial_actions)

        joint_spatial_actions = tf.add(
            spatial_actions_normal,
            previous_spatial_actions_added,
            'spatial_actions_add'
        )

        if self.trainable:
            tf.summary.image(f"spatial_action_normal", tf.reshape(spatial_actions_normal, [-1, 32, 32, 1]), 3)
            tf.summary.image(f"spatial_action_previous", tf.reshape(spatial_actions_previous, [-1, 32, 32, 1]), 3)
            tf.summary.image(f"joint_connected_layers", tf.reshape(joint_spatial_actions, [-1, 32, 32, 1]), 3)

        # Take the softmax of this final convolutional layer.
        spatial_action_probs = tf.nn.softmax(layers.flatten(joint_spatial_actions))

        # Build a full connected layer of this final convolutional layer.
        # Could possibly pass in additional variables here, alongside the
        # convolutional layer.
        map_output_flat = layers.flatten(visual_inputs)

        fully_connected_layer_normal = layers.fully_connected(
            map_output_flat,
            num_outputs=256,
            activation_fn=None,
            scope=f"fully_connected_layer1/model_{self.curriculum_number}",
            trainable=self.trainable
        )

        previous_fully_con_1 = []
        for model_number, prev_out in enumerate(previous_model.flatten_1):
            fully_connected_previous = layers.fully_connected(
                prev_out,
                num_outputs=256,
                activation_fn=None,
                scope=f"fully_connected_layer1/model_{model_number}",
                trainable=self.trainable
            )

            previous_fully_con_1.append(fully_connected_previous)

        previous_fully_con_1_added = self.add_all_previous(previous_fully_con_1)

        # Combine the new and old models values, and then apply RELU to the result.
        joint_connected_layers = tf.add(
            fully_connected_layer_normal,
            previous_fully_con_1_added,
            'fully_connected_layer_add'
        )

        relu_connected_layer = tf.nn.relu(
            joint_connected_layers,
            name='fully_connected_layer1_normal_relu'
        )

        # Generate the probability of a given action from the
        # fully connected layer. Finally, produce a value
        # estimate for the given actions.
        action_id_probs_new = layers.fully_connected(
            relu_connected_layer,
            num_outputs=len(actions.FUNCTIONS),
            activation_fn=None,
            scope=f"action_id/model_{self.curriculum_number}",
            trainable=self.trainable
        )

        previous_action_ids = []
        for model_number, prev_out in enumerate(previous_model.fully_connected_layer1):
            previous_action_id_probs = layers.fully_connected(
                prev_out,
                num_outputs=len(actions.FUNCTIONS),
                activation_fn=None,
                scope=f"action_id/model_{model_number}",
                trainable=self.trainable
            )

            previous_action_ids.append(previous_action_id_probs)

        previous_action_ids_added = self.add_all_previous(previous_action_ids)

        joint_action_ids = tf.add(
            action_id_probs_new,
            previous_action_ids_added,
            'id_probs_add'
        )

        # Combine the new and old models values, and then apply softmax to the result.
        action_id_probs = tf.nn.softmax(joint_action_ids)

        # Sort value estimate.
        value_estimate_new = layers.fully_connected(
            relu_connected_layer,
            num_outputs=1,
            activation_fn=None,
            scope=f"value/model_{self.curriculum_number}",
            trainable=self.trainable
        )

        previous_value_estimates = []
        for model_number, prev_out in enumerate(previous_model.fully_connected_layer1):
            value_estimate_previous = layers.fully_connected(
                prev_out,
                num_outputs=1,
                activation_fn=None,
                scope=f"value/model_{model_number}",
                trainable=self.trainable
            )

            previous_value_estimates.append(value_estimate_previous)

        previous_value_estimates_added = self.add_all_previous(previous_value_estimates)

        # Combine the new and old models values, and then squeeze the result.
        joint_value_estimate = tf.add(
            value_estimate_new,
            previous_value_estimates_added,
            'value_estimate_add'
        )

        value_estimate = tf.squeeze(
            joint_value_estimate,
            axis=1
        )

        # Disregard all the non-allowed actions by giving them a
        # probability of zero, before re-normalizing to 1.
        action_id_probs *= self.placeholders.available_action_ids
        action_id_probs /= tf.reduce_sum(action_id_probs, axis=1, keepdims=True)

        spatial_action_log_probs = (
            self.logclip(spatial_action_probs)
            * tf.expand_dims(self.placeholders.is_spatial_action_available, axis=1)
        )

        action_id_log_probs = self.logclip(action_id_probs)

        self.value_estimate = value_estimate
        self.action_id_probs = action_id_probs
        self.spatial_action_probs = spatial_action_probs
        self.action_id_log_probs = action_id_log_probs
        self.spatial_action_log_probs = spatial_action_log_probs

        return self
