import os
import shutil
import signal
import sys

import tensorflow as tf

from datetime import datetime
from functools import partial
from absl import flags

from agent.agent import A2C
from agent.runner import Runner
from common.multienv import SubprocVecEnv, make_sc2env

from pysc2.env import sc2_env

# Flags taken from example code at https://github.com/xhujoy/pysc2-agents/blob/master/main.py

FLAGS = flags.FLAGS
flags.DEFINE_boolean("training", True, "Should the agent be trained.")
flags.DEFINE_boolean("visualize", False, "Whether to render with PyGame.")

flags.DEFINE_integer("resolution", 32, "Resolution for screen and mini-map feature layers.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")
flags.DEFINE_integer("n_envs", 1, "Number of environments to run in parallel.")
flags.DEFINE_integer("n_steps_per_batch", None, "Number of steps per batch, if None use 8.")
flags.DEFINE_integer("all_summary_freq", 50, "Record all summaries every n batch.")
flags.DEFINE_integer("scalar_summary_freq", 5, "Record scalar summaries every n batch.")
flags.DEFINE_integer("K_batches", -1, "Number of training in thousands, -1 to run forever.")
flags.DEFINE_integer("save_replays_every", 0, "How often to save a replay, 0 for never.")

flags.DEFINE_string("checkpoint_path", "_files/models", "Path for agent checkpoints.")
flags.DEFINE_string("summary_path", "_files/summaries", "Path for tensorboard summaries.")
flags.DEFINE_string("model_name", "temp_testing", "Name for checkpoints and tensorboard summaries.")
flags.DEFINE_string("map_name", "MoveToBeacon", "Map to use.")
flags.DEFINE_string("replay_dir", "", "Where to say the replays to.")

flags.DEFINE_enum("if_output_exists", "fail", ["fail", "overwrite", "continue"],
                  "What to do if output exists, only for training, is ignored if not training.")

flags.DEFINE_float("discount", 0.95, "Reward-discount for the agent.")
flags.DEFINE_float("max_gradient_norm", 500.0, "The maximum gradient norm.")
flags.DEFINE_float("loss_value_weight", 1.0, "Weight for the Loss Function.")
flags.DEFINE_float("entropy_weight_spatial", 1e-6, "Entropy of spatial action distribution loss weight.")
flags.DEFINE_float("entropy_weight_action", 1e-6, "Entropy of action-id distribution loss weight.")

FLAGS(sys.argv)

FULL_CHECKPOINT_PATH = os.path.join(FLAGS.checkpoint_path, FLAGS.model_name)
HAVE_BEEN_KILLED = False

if FLAGS.training:
    FULL_SUMMARY_PATH = os.path.join(FLAGS.summary_path, FLAGS.model_name)
else:
    FULL_SUMMARY_PATH = os.path.join(FLAGS.summary_path, "no_training", FLAGS.model_name)

def signal_term_handler(signal, frame):
    """signal_term_handler

    Process the SIGTERM event by setting a global var to gracefully save and close
    """

    print("Received a SIGTERM command, closing!")
    HAVE_BEEN_KILLED = True


signal.signal(signal.SIGTERM, signal_term_handler)


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
    """main

    Main function of the CNN code.
    Given the input flags, setup the agent
    and its runner, before starting the main
    running loop.
    """

    # Check the save folders, before creating a
    # dictionary of the running parameters to be passed
    # to the environment setup function.
    if FLAGS.training:
        check_existing_folder(FULL_CHECKPOINT_PATH)
        check_existing_folder(FULL_SUMMARY_PATH)

    if FLAGS.save_replays_every > 0:
        if FLAGS.replay_dir == "":
            print("Need to specify a replay dir!")
            return

    agent_details = sc2_env.AgentInterfaceFormat(
        feature_dimensions=sc2_env.Dimensions(
            screen=(FLAGS.resolution,) * 2,
            minimap=(FLAGS.resolution,) * 2
        )
    )

    environment_arguments = dict(
        map_name=FLAGS.map_name,
        step_mul=FLAGS.step_mul,
        game_steps_per_episode=0,
        #screen_size_px=(FLAGS.resolution,) * 2,
        #minimap_size_px=(FLAGS.resolution,) * 2,
        visualize=FLAGS.visualize,
        save_replay_episodes=FLAGS.save_replays_every,
        replay_dir=FLAGS.replay_dir,
        agent_interface_format=agent_details
    )

    environment = SubprocVecEnv(
        (partial(make_sc2env, **environment_arguments),) * FLAGS.n_envs
    )

    # Setup the agent and its runner.
    tf.reset_default_graph()
    session = tf.Session()

    agent = A2C(
        session=session,
        spatial_dim=FLAGS.resolution,
        unit_type_emb_dim=5,
        loss_value_weight=FLAGS.loss_value_weight,
        entropy_weight_action_id=FLAGS.entropy_weight_action,
        entropy_weight_spatial=FLAGS.entropy_weight_spatial,
        scalar_summary_freq=FLAGS.scalar_summary_freq,
        summary_path=FULL_SUMMARY_PATH,
        all_summary_freq=FLAGS.all_summary_freq,
        max_gradient_norm=FLAGS.max_gradient_norm
    )

    agent.build_model()

    # If there is a checkpoint, we should load it.
    if os.path.exists(FULL_CHECKPOINT_PATH):
        agent.load(FULL_CHECKPOINT_PATH)
    else:
        agent.init()

    # Check that number of steps per batch is defined.
    if FLAGS.n_steps_per_batch is None:
        n_steps_per_batch = 8
    else:
        n_steps_per_batch = FLAGS.n_steps_per_batch

    # Setup the runner.
    runner = Runner(
        envs=environment,
        agent=agent,
        discount=FLAGS.discount,
        n_steps=n_steps_per_batch,
        do_training=FLAGS.training,
    )

    runner.reset()

    if FLAGS.K_batches >= 0:
        n_batches = FLAGS.K_batches * 1000
    else:
        n_batches = -1

    current_iter = 0

    # Run training until the user interrupts or the
    # number of batches is met.
    # Log the current iteration every 500 batches,
    # and save every 2000.
    try:
        while True:

            if HAVE_BEEN_KILLED:
                break

            if current_iter % 500 == 0:
                print_and_log(current_iter)
                save(agent)

            runner.run_batch()

            current_iter += 1

            if 0 <= n_batches <= current_iter:
                break
    except KeyboardInterrupt:
        pass

    # Save, clean up and quit.
    print_and_log(current_iter)
    save(agent)

    print("Finished saving")

    environment.close()


if __name__ == "__main__":
    main()
