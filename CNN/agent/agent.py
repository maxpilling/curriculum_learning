import collections
import os

import numpy as np
import tensorflow as tf

from agent.policy import ConvPolicy
from common.preprocess import ObsProcessor, FEATURE_KEYS, AgentInputTuple
from common.util import weighted_random_sample, select_from_each_row, ravel_index_pairs
from pysc2.lib import actions
from tensorflow.contrib import layers
from tensorflow.contrib.layers.python.layers.optimizers import OPTIMIZER_SUMMARIES

# A named tuple to store the selected probabilities together.
SelectedLogProbs = collections.namedtuple("SelectedLogProbs", ["action_id", "spatial", "total"])

def get_default_values(spatial_dimensions):
    """get_default_values

    The initial values for the TensorFlow features.

    :param spatial_dimensions: The size of the spatial dimension.
    """

    s_d = spatial_dimensions

    feature_list = [
        (
            FEATURE_KEYS.minimap_numeric,
            tf.float32,
            [None, s_d, s_d, ObsProcessor.N_MINIMAP_CHANNELS]
        ),
        (
            FEATURE_KEYS.screen_numeric,
            tf.float32,
            [None, s_d, s_d, ObsProcessor.N_SCREEN_CHANNELS]
        ),
        (
            FEATURE_KEYS.non_spatial_features,
            tf.float32,
            [None, ObsProcessor.N_NON_SPATIAL]
        ),
        (
            FEATURE_KEYS.screen_unit_type,
            tf.int32,
            [None, s_d, s_d]
        ),
        (
            FEATURE_KEYS.is_spatial_action_available,
            tf.float32,
            [None]
        ),
        (
            FEATURE_KEYS.available_action_ids,
            tf.float32,
            [None, len(actions.FUNCTIONS)]
        ),
        (
            FEATURE_KEYS.selected_spatial_action,
            tf.int32,
            [None, 2]
        ),
        (
            FEATURE_KEYS.selected_action_id,
            tf.int32,
            [None]
        ),
        (
            FEATURE_KEYS.value_target,
            tf.float32,
            [None]
        ),
        (
            FEATURE_KEYS.player_relative_screen,
            tf.int32,
            [None, s_d, s_d]
        ),
        (
            FEATURE_KEYS.player_relative_minimap,
            tf.int32,
            [None, s_d, s_d]
        ),
        (
            FEATURE_KEYS.advantage,
            tf.float32,
            [None]
        )
    ]

    return AgentInputTuple(
        **{name: tf.placeholder(dtype, shape, name) for name, dtype, shape in feature_list}
    )

class A2C:

    _scalar_summary_key = "scalar_summaries"

    def __init__(self,
                 session: tf.Session,
                 summary_path: str,
                 all_summary_freq: int,
                 scalar_summary_freq: int,
                 spatial_dim: int,
                 unit_type_emb_dim=4,
                 loss_value_weight=1.0,
                 entropy_weight_spatial=1e-6,
                 entropy_weight_action_id=1e-5,
                 max_gradient_norm=None,
                 optimiser_params=None,
                ):
        """
        Convolutional Based Agent for learning PySC2 Mini-games
        Tidied and altered code from https://github.com/pekaalto/sc2aibot

        :param session: The TensorFlow session to be used.
        :param summary_path: Path to store TensorFlow summaries in.
        :param all_summary_freq: How often to save all summaries.
        :param scalar_summary_freq: How often save scalar summaries.
        :param spatial_dim: Dimension for both the mini-map and the screen.
        :param unit_type_emb_dim: The unit type embedding dimensions, used for mappings.
        :param loss_value_weight: Value weight for the update step.
        :param entropy_weight_spatial: Spatial entropy for update step.
        :param entropy_weight_action_id: Action selection entropy for update step.
        :param max_gradient_norm: Max norm for gradients, if None then no limit.
        :param optimiser_params: Parameters to be passed to the optimiser.
        """

        # Setup the values passed over.
        self.session = session
        self.spatial_dim = spatial_dim
        self.loss_value_weight = loss_value_weight
        self.entropy_weight_spatial = entropy_weight_spatial
        self.entropy_weight_action_id = entropy_weight_action_id
        self.unit_type_emb_dim = unit_type_emb_dim
        self.summary_path = summary_path

        # Check the path is okay before pointing TensorFlow to it.
        os.makedirs(summary_path, exist_ok=True)

        self.summary_writer = tf.summary.FileWriter(summary_path)
        self.all_summary_freq = all_summary_freq
        self.scalar_summary_freq = scalar_summary_freq
        self.train_step = 0
        self.max_gradient_norm = max_gradient_norm
        self.policy = ConvPolicy

        opt_class = tf.train.AdamOptimizer

        # Use some default values if none are passed over.
        if optimiser_params != None:
            params = optimiser_params
        else:
            params = {
                "learning_rate": 1e-4,
                "epsilon": 5e-7
            }

        self.optimiser = opt_class(**params)

    def init(self):
        """init

        Start the session with the initial operation.
        """

        self.session.run(self.init_op)

    def get_scalar_summary(self, name, tensor):
        """get_scalar_summary

        Wrapper function for the `tf.summary.scalar` function

        :param name: Name of the value.
        :param tensor: The tensor to summarise.
        """
        tf.summary.scalar(name,
                          tensor,
                          collections=[tf.GraphKeys.SUMMARIES, self._scalar_summary_key])

    def get_selected_action_probability(self, theta, selected_spatial_action):
        """get_selected_action_probability

        :param theta: The current state of the policy.
        :param selected_spatial_action: The current spatial action to evaluate.
        """

        action_id = select_from_each_row(
            theta.action_id_log_probs, self.placeholders.selected_action_id
        )

        spatial_coord = select_from_each_row(
            theta.spatial_action_log_probs, selected_spatial_action
        )

        total = spatial_coord + action_id

        return SelectedLogProbs(action_id, spatial_coord, total)

    def build_model(self):
        """build_model

        Function that actually builds the model, initialising
        variables and setting up the policy.
        After this, it sets up the loss value, defines a training
        step and sets up logging for all needed values.
        """

        # Initialise the placeholders property with some default values.
        self.placeholders = get_default_values(self.spatial_dim)

        # Provides checks to ensure that variable isn't shared by accident,
        # and starts up the fully convolutional policy.
        with tf.variable_scope("theta"):
            theta = self.policy(
                self,
                trainable=True,
                spatial_dim=self.spatial_dim
            ).build()

        # Get the actions and the probabilities of those actions.
        selected_spatial_action = ravel_index_pairs(
            self.placeholders.selected_spatial_action, self.spatial_dim
        )

        selected_log_probabilities = self.get_selected_action_probability(
            theta,
            selected_spatial_action
        )

        # Take the maximum here to avoid a divide by 0 error next.
        sum_of_available_spatial = tf.maximum(
            1e-10,
            tf.reduce_sum(self.placeholders.is_spatial_action_available)
        )

        # Generate the negative entropy, used later as part of the loss
        # function. This in-turn is used to optimise to get the lowest
        # loss possible.
        negative_spatial_entropy = tf.reduce_sum(
            theta.spatial_action_probs * theta.spatial_action_log_probs
        )

        negative_spatial_entropy /= sum_of_available_spatial

        negative_entropy_for_action_id = tf.reduce_mean(
            tf.reduce_sum(
                theta.action_id_probs * theta.action_id_log_probs,
                axis=1
            )
        )

        # Get the values for the possible actions.
        self.sampled_action_id = weighted_random_sample(theta.action_id_probs)
        self.sampled_spatial_action = weighted_random_sample(theta.spatial_action_probs)

        self.value_estimate = theta.value_estimate

        # Calculate the policy and value loss, such that the final loss
        # can be calculated and optimised against.
        policy_loss = - tf.reduce_mean(
            selected_log_probabilities.total * self.placeholders.advantage
        )

        value_loss = tf.losses.mean_squared_error(
            self.placeholders.value_target,
            theta.value_estimate
        )

        total_loss = (
            policy_loss +
            value_loss * self.loss_value_weight +
            negative_spatial_entropy * self.entropy_weight_spatial +
            negative_entropy_for_action_id * self.entropy_weight_action_id
        )

        # Define a training step to be optimising the loss to be the lowest.
        self.train_operation = layers.optimize_loss(
            loss=total_loss,
            global_step=tf.train.get_global_step(),
            optimizer=self.optimiser,
            clip_gradients=self.max_gradient_norm,
            summaries=OPTIMIZER_SUMMARIES,
            learning_rate=None,
            name="train_operation"
        )

        # Finally, log some information about the model in its current state.
        self.get_scalar_summary(
            "Value - Estimate:",
            tf.reduce_mean(self.value_estimate)
        )

        self.get_scalar_summary(
            "Value - Target:",
            tf.reduce_mean(self.placeholders.value_target)
        )

        self.get_scalar_summary(
            "Action - Is Spatial Action Available:",
            tf.reduce_mean(self.placeholders.is_spatial_action_available)
        )

        self.get_scalar_summary(
            "Action - Selected Action ID Log Probability",
            tf.reduce_mean(selected_log_probabilities.action_id)
        )

        self.get_scalar_summary("Loss - Policy Loss", policy_loss)
        self.get_scalar_summary("Loss - Value Loss", value_loss)
        self.get_scalar_summary("Loss - Negative Spatial Entropy", negative_spatial_entropy)
        self.get_scalar_summary(
            "Loss - Negative Entropy for Action ID",
            negative_entropy_for_action_id
        )

        self.get_scalar_summary("Loss - Total", total_loss)
        self.get_scalar_summary(
            "Value - Advantage",
            tf.reduce_mean(self.placeholders.advantage)
        )

        self.get_scalar_summary(
            "Action - Selected Total Log Probability",
            tf.reduce_mean(selected_log_probabilities.total)
        )

        self.get_scalar_summary(
            "Action - Selected Spatial Action Log Probability",
            tf.reduce_sum(selected_log_probabilities.spatial) / sum_of_available_spatial
        )

        # Clean up and save.
        self.init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver(max_to_keep=2)
        self.all_summary_op = tf.summary.merge_all(tf.GraphKeys.SUMMARIES)
        self.scalar_summary_op = tf.summary.merge(
            tf.get_collection(self._scalar_summary_key)
        )

    @staticmethod
    def organise_obs_for_session(obs):
        """organise_obs_for_session

        Nicely format the Obs object, such that it can
        be fed to the running or training of the model.

        :param obs: The observation object, passed from the SC2LE.
        """

        return {k + ":0": v for k, v in obs.items()}

    def step(self, obs):
        """step

        Take a step of the CNN, i.e. generate an
        Action, Coord, Value tuple, based on the current
        game state.

        :param obs: The observation object, passed from the SC2LE.
        """

        # Format the obs object nicely, before feeding it to running the current
        # session.
        feed_dict = self.organise_obs_for_session(obs)

        # Generate an Action Id, the spatial representation and the value
        # estimate for that pair.
        action_to_take, spatial_action, value_estimate = self.session.run(
            [self.sampled_action_id, self.sampled_spatial_action, self.value_estimate],
            feed_dict=feed_dict
        )

        # Swap back to an actual 2D grid.
        spatial_action_2d = np.array(
            np.unravel_index(spatial_action, (self.spatial_dim,) * 2)
        )

        # Flip for the backwards co-ordinate system.
        spatial_action_2d = spatial_action_2d.transpose()

        return action_to_take, spatial_action_2d, value_estimate

    def train(self, obs):
        """train

        Take a step of training, to improve the CNN model.

        :param obs: The observation object, passed from the SC2LE.
        """

        feed_dict = self.organise_obs_for_session(obs)

        operations = [self.train_operation]

        # If either of the summaries need writing, this
        # should be added to the current operations.
        should_write_all_summaries = (
            (self.train_step % self.all_summary_freq == 0) and
            self.summary_path is not None
        )

        should_write_scalar_summaries = (
            (self.train_step % self.scalar_summary_freq == 0) and
            self.summary_path is not None
        )

        if should_write_all_summaries:
            operations.append(self.all_summary_op)
        elif should_write_scalar_summaries:
            operations.append(self.scalar_summary_op)

        # Run and log the values as summaries if needed,
        # before incrementing the current training step.
        run_value = self.session.run(operations, feed_dict)

        if should_write_all_summaries or should_write_scalar_summaries:
            self.summary_writer.add_summary(run_value[-1], global_step=self.train_step)

        self.train_step += 1

    def get_value(self, obs):
        """get_value

        Helper function to get the value for a given
        state of observations.

        :param obs: The observation object, passed from the SC2LE.
        """

        feed_dict = self.organise_obs_for_session(obs)

        return self.session.run(
            self.value_estimate,
            feed_dict=feed_dict
        )

    def flush_summaries(self):
        """flush_summaries

        Simple helper function to flush the summary writer,
        such that the runner can ensure the summary log has been
        fully cleared.
        """

        self.summary_writer.flush()

    def save(self, path, step=None):
        """save

        Save the model to the given path,
        for later re-use.

        :param path: The path to save to.
        :param step: The optional step the model is on.
        """

        os.makedirs(path, exist_ok=True)

        step = step or self.train_step
        print("Saving the model to %s, at step %d" % (path, step))
        self.summary_writer.add_graph(self.session.graph)
        self.saver.save(
            self.session,
            path + '/model.ckpt',
            global_step=step
        )

    def load(self, path):
        """load

        Load an older model for re-use from a given path.

        :param path: The path to load the model from.
        """

        checkpoint = tf.train.get_checkpoint_state(path)

        self.saver.restore(self.session, checkpoint.model_checkpoint_path)

        # Reload the training step the loaded model was at.
        self.train_step = int(checkpoint.model_checkpoint_path.split('-')[-1])

        print("Loaded old model with training step: %d" % self.train_step)

        # Now increment, since we are on the next step.
        self.train_step += 1
