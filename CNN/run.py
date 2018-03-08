import os
import shutil
import sys

import tensorflow as tf

from datetime import datetime
from functools import partial
from absl import flags

from actorcritic.agent import ActorCriticAgent, ACMode
from actorcritic.runner import Runner, PPORunParams
from common.multienv import SubprocVecEnv, make_sc2env

# Flags taken from example code at https://github.com/xhujoy/pysc2-agents/blob/master/main.py

FLAGS = flags.FLAGS
flags.DEFINE_boolean("training", True, "Should the agent be trained.")
flags.DEFINE_bool("visualize", False, "Whether to render with PyGame.")
flags.DEFINE_integer("resolution", 32, "Resolution for screen and mini-map feature layers.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")
flags.DEFINE_integer("n_envs", 1, "Number of environments to run in parallel.")
flags.DEFINE_integer("n_steps_per_batch", None, "Number of steps per batch, if None use 8.")
flags.DEFINE_integer("all_summary_freq", 50, "Record all summaries every n batch.")
flags.DEFINE_integer("scalar_summary_freq", 5, "Record scalar summaries every n batch.")
flags.DEFINE_string("checkpoint_path", "_files/models", "Path for agent checkpoints.")
flags.DEFINE_string("summary_path", "_files/summaries", "Path for tensorboard summaries.")
flags.DEFINE_string("model_name", "temp_testing", "Name for checkpoints and tensorboard summaries.")
flags.DEFINE_integer("K_batches", -1, "Number of training in thousands, -1 to run forever.")
flags.DEFINE_string("map_name", "MoveToBeacon", "Map to use.")
flags.DEFINE_float("discount", 0.95, "Reward-discount for the agent.")
flags.DEFINE_enum("if_output_exists", "fail", ["fail", "overwrite", "continue"],
                  "What to do if summary and model output exists, only for training, is ignored if not training.")
flags.DEFINE_float("max_gradient_norm", 500.0, "The maximum gradient norm.")
flags.DEFINE_float("loss_value_weight", 1.0, "Weight for the Loss Function.")

flags.DEFINE_float("entropy_weight_spatial", 1e-6, "Entropy of spatial action distribution loss weight.")
flags.DEFINE_float("entropy_weight_action", 1e-6, "Entropy of action-id distribution loss weight.")

FLAGS(sys.argv)

FULL_CHECKPOINT_PATH = os.path.join(FLAGS.checkpoint_path, FLAGS.model_name)

if FLAGS.training:
    FULL_SUMMARY_PATH = os.path.join(FLAGS.summary_path, FLAGS.model_name)
else:
    FULL_SUMMARY_PATH = os.path.join(FLAGS.summary_path, "no_training", FLAGS.model_name)

def check_existing_folder(folder):
    """check_existing_folder

    Check if the supplied folder exists, and remove if suitable.

    :param folder: The folder to check.
    """
    if os.path.exists(folder):
        if FLAGS.if_output_exists == "overwrite":
            shutil.rmtree(folder)
            print(f"Removed old model from {folder}.")
        elif FLAGS.if_output_exists == "fail":
            raise Exception(f"Model {folder} already exists." + 
                            " Set --if_output_exists to 'overwrite' to delete.")


def print_and_log(data):
    """print_and_log

    Print the current batch info, and also log to a log file.

    :param data: The data to print.
    """
    print(datetime.now())
    print(f"Batch number: {data}")
    sys.stdout.flush()


def save(agent):
    """save

    Save the agent if training mode is enabled.

    :param agent: The agent to save.
    """
    if FLAGS.training:
        agent.save(FULL_CHECKPOINT_PATH)
        agent.flush_summaries()
        sys.stdout.flush()

def main():

    if FLAGS.training:
        check_existing_folder(FULL_CHECKPOINT_PATH)
        check_existing_folder(FULL_SUMMARY_PATH)

    environment_arguments = dict(
        map_name=FLAGS.map_name,
        step_mul=FLAGS.step_mul,
        game_steps_per_episode=8,
        screen_size_px=(FLAGS.resolution,) * 2,
        minimap_size_px=(FLAGS.resolution,) * 2,
        visualize=FLAGS.visualize
    )

if __name__ == "__main__":
    main()
