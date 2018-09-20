"""util

Various utility functions.
"""

import numpy as np
import tensorflow as tf


def weighted_random_sample(weights):
    """weighted_random_sample

    Given some weights, return a 1D tensor with the ids randomly
    sampled, proportional to the weights.

    :param weights: 2D tensor [n, d] containing positive weights for sampling.
    """

    uniform = tf.random_uniform(tf.shape(weights))
    return tf.argmax(tf.log(uniform) / weights, axis=1)


def select_from_each_row(params, indices):
    """select_from_each_row

    Given a 2D tensor an accompanying 1D tensor, return
    a 1D tensor where each value is the associated
    value from that row, as pointed to by the indices tensor.

    :param params: 2D tensor of shape [d1,d2].
    :param indices: 1D tensor of shape [d1] with values in [d1, d2].
    """

    sel = tf.stack([tf.range(tf.shape(params)[0]), indices], axis=1)
    return tf.gather_nd(params, sel)


def calculate_n_step_reward(
    one_step_rewards: np.ndarray, discount: float, last_state_values: np.ndarray
):
    """calculate_n_step_reward

    Given the one step rewards, the discount and the last values,
    return the cumulative sum of the full rewards, with discount
    applied.

    :param one_step_rewards: [n_env, n_timesteps].
    :param discount: Scalar discount paramater.
    :param last_state_values: [n_env], bootstrap from these if not done.
    """

    discount = discount ** np.arange(one_step_rewards.shape[1], -1, -1)
    reverse_rewards = np.c_[one_step_rewards, last_state_values][:, ::-1]
    full_discounted_reverse_rewards = reverse_rewards * discount
    return (np.cumsum(full_discounted_reverse_rewards, axis=1) / discount)[:, :0:-1]


def general_n_step_advantage(
    one_step_rewards: np.ndarray,
    value_estimates: np.ndarray,
    discount: float,
    lambda_par: float,
):
    """general_n_step_advantage

    Get the general n step advantage.

    :param one_step_rewards: [n_env, n_timesteps]
    :param value_estimates: [n_env, n_timesteps + 1]
    :param discount: "gamma" in https://arxiv.org/pdf/1707.06347.pdf and most of the rl-literature
    :param lambda_par: "lambda" in https://arxiv.org/pdf/1707.06347.pdf
    """

    # Check the inputs are valid.
    assert 0.0 < discount <= 1.0
    assert 0.0 <= lambda_par <= 1.0

    batch_size, timesteps = one_step_rewards.shape

    assert value_estimates.shape == (batch_size, timesteps + 1)

    delta = (
        one_step_rewards + discount * value_estimates[:, 1:] - value_estimates[:, :-1]
    )

    if lambda_par == 0:
        return delta

    delta_rev = delta[:, ::-1]
    adjustment = (discount * lambda_par) ** np.arange(timesteps, 0, -1)
    advantage = (np.cumsum(delta_rev * adjustment, axis=1) / adjustment)[:, ::-1]

    return advantage


def combine_first_dimensions(input_arr: np.ndarray):
    """combine_first_dimensions

    Combines the first dimension of a given array.
    This is usually batch_size * time.

    :param x: array of [batch_size, time, ...]
    """

    first_dim = input_arr.shape[0] * input_arr.shape[1]
    other_dims = input_arr.shape[2:]
    dims = (first_dim,) + other_dims
    return input_arr.reshape(*dims)


def ravel_index_pairs(idx_pairs, n_col):
    """ravel_index_pairs

    Reduce a pair of indicies.
    """

    return tf.reduce_sum(idx_pairs * np.array([n_col, 1])[np.newaxis, ...], axis=1)


def dict_of_lists_to_list_of_dicts(dict_of_lists: dict):
    """dict_of_lists_to_list_of_dicts

    Given an input dict of lists, convert it to a list of
    dictionaries.

    :param x: The input dictionary.
    :type x: dict
    """

    dim = {len(v) for v in dict_of_lists.values()}

    assert len(dim) == 1

    dim = dim.pop()

    return [{k: dict_of_lists[k][i] for k in dict_of_lists} for i in range(dim)]


def dump_all_tensors_to_file(graph, path):
    """dump_all_tensors_to_file

    Dumps all of the tensors for a given graph to a new file.
    """

    nodes = [n.name for n in graph.as_graph_def().node]
    with open(path, "w") as f:
        for node in nodes:
            print(node, file=f)
