import tensorflow as tf
from math import ceil

DEBUG = True

def pad_and_tile_non_spatial(conv_policy, log_non_spatial_features):
    """pad_and_tile_non_spatial

    Pad the 2D matrix with zeros.
    Tile the 3D matrix with the 2D matrix.
    """

    if DEBUG:
        print("Using Pad and Tile")

    non_spatial_dim_size = conv_policy.placeholders.non_spatial_features.get_shape().as_list()[1]
    size_difference = conv_policy.spatial_dim - non_spatial_dim_size

    padding = tf.constant([[0, 0], [0, size_difference]])
    
    non_spatial_padded = tf.pad(
        log_non_spatial_features,
        padding,
        'CONSTANT'
    )

    if DEBUG:
        print(f"Two D shape: ({non_spatial_padded.get_shape().as_list()})")

    three_d_non_spatial_init = tf.expand_dims(non_spatial_padded, 2) 
    tiles = [1, 1, conv_policy.spatial_dim]

    three_d_non_spatial = tf.tile(
        three_d_non_spatial_init,
        tiles
    )

    if DEBUG:
        print(f"Three D shape: ({three_d_non_spatial.get_shape().as_list()})")
        print(f"Screen shape: ({tf.shape(conv_policy.placeholders.screen_numeric)[0]})")

    four_d_non_spatial = tf.expand_dims(
        three_d_non_spatial,
        3
    )

    if DEBUG:
        print(f"Four D final shape: ({four_d_non_spatial.get_shape().as_list()})")

    return four_d_non_spatial

def tile_and_tile_non_spatial(conv_policy, log_non_spatial_features):
    """tile_and_tile_non_spatial

    Tile the 2D matrix with the initial vector.
    Tile the 3D matrix with the 2D matrix.
    """

    if DEBUG:
        print("Using Tile and Tile")

    non_spatial_dim_size = conv_policy.placeholders.non_spatial_features.get_shape().as_list()[1]

    tile_number = ceil(conv_policy.spatial_dim / non_spatial_dim_size)
    size_difference = (tile_number * non_spatial_dim_size) - conv_policy.spatial_dim
    tiles = [1, tile_number]

    non_spatial_tiled_init = tf.tile(
        log_non_spatial_features,
        tiles
    )

    non_spatial_tiled = non_spatial_tiled_init[:, :conv_policy.spatial_dim]

    if DEBUG:
        print(f"Two D shape: ({non_spatial_tiled.get_shape().as_list()})")

    three_d_non_spatial_init = tf.expand_dims(non_spatial_tiled, 2) 
    tiles = [1, 1, 32]

    three_d_non_spatial = tf.tile(
        three_d_non_spatial_init,
        tiles
    )

    if DEBUG:
        print(f"Three D shape: ({three_d_non_spatial.get_shape().as_list()})")
        print(f"Screen shape: ({tf.shape(conv_policy.placeholders.screen_numeric)[0]})")

    four_d_non_spatial = tf.expand_dims(
        three_d_non_spatial,
        3
    )

    if DEBUG:
        print(f"Four D final shape: ({four_d_non_spatial.get_shape().as_list()})")

    return four_d_non_spatial