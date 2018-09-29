import sys

import numpy as np
import tensorflow as tf

from agent.agent import A2C
from absl import flags
from collections import namedtuple
from common.preprocess import ObsProcessor, ActionProcessor, FEATURE_KEYS
from common.util import general_n_step_advantage, combine_first_dimensions


class Runner(object):
    """Runner

    Helper class to manage the running of the agent.
    """

    def __init__(
        self,
        envs,
        agent: A2C,
        n_steps=5,
        discount=0.99,
        do_training=True,
        number_episodes=-1,
        episode_counter=0
    ):
        self.envs = envs
        self.agent = agent
        self.obs_processor = ObsProcessor()
        self.action_processor = ActionProcessor(dim=flags.FLAGS.resolution)
        self.n_steps = n_steps
        self.discount = discount
        self.do_training = do_training
        self.batch_counter = 0
        self.episode_counter = episode_counter
        self.number_episodes = number_episodes
        self.current_step = 0

    def reset(self):
        """reset

        Process the reset of the runner, setting the latest observation back
        to some initial state.
        """

        obs = self.envs.reset()
        self.latest_obs = self.obs_processor.process(obs)

    def _log_score_to_tb(self, score):
        """_log_score_to_tb

        Pass a score over to the summary writer in the agent for logging.

        :param score: Score to log.
        """

        summary = tf.Summary()
        summary.value.add(tag="sc2/episode_score", simple_value=score)
        self.agent.summary_writer.add_summary(summary, self.episode_counter)

    def _handle_episode_end(self, timestep):
        """_handle_episode_end

        Deal with an episode ending, by logging the score and increasing
        the stored episode number.

        :param timestep: The timestep the episode ended at.
        """

        # Get the score at the end of the episode.
        # Critically, this uses the "score_cumulative" which
        # is only defined in the mini-games, so would need updating
        # if used for a different type of scoring system.
        score = timestep.observation["score_cumulative"][0]
        current_steps = timestep.observation["game_loop"][0]
        print(f"Episode {self.episode_counter} ended. Score {score} at step {current_steps}")

        self._log_score_to_tb(score)
        self.episode_counter += 1

    def run_batch(self):
        """run_batch

        Run a batch of the training, building up a list of actions, observations,
        values of those actions and the rewards given.
        """

        if self.number_episodes != -1 and self.number_episodes <= self.episode_counter:
            print("Max number episodes reached. Quitting.")
            return False

        # Define variables to store the actions, observations, values and rewards in.
        mb_actions = []
        mb_obs = []
        mb_values = np.zeros((self.envs.n_envs, self.n_steps + 1), dtype=np.float32)
        mb_rewards = np.zeros((self.envs.n_envs, self.n_steps), dtype=np.float32)

        latest_obs = self.latest_obs

        # For the number of steps, save the relevant data each step.
        # When finished, deal with the episode end.
        for n_step in range(self.n_steps):
            # Save the value estimate here, to make the n step reward calculation easier.
            action_id, spatial_action_2d, value_estimate = self.agent.step(latest_obs)

            mb_values[:, n_step] = value_estimate
            mb_obs.append(latest_obs)
            mb_actions.append((action_id, spatial_action_2d))

            actions_pp = self.action_processor.process(action_id, spatial_action_2d)
            obs_raw = self.envs.step(actions_pp)
            latest_obs = self.obs_processor.process(obs_raw)
            mb_rewards[:, n_step] = [t.reward for t in obs_raw]
            self.current_step += 1

            for timestep in obs_raw:
                if timestep.last():
                    self._handle_episode_end(timestep)
            print(f"Total game steps: {self.agent.get_training_step()}")

        mb_values[:, -1] = self.agent.get_value(latest_obs)

        n_step_advantage = general_n_step_advantage(
            mb_rewards, mb_values, self.discount, lambda_par=1.0
        )

        full_input = {
            # These are transposed because action/obs
            # processors return [time, env, ...] shaped arrays.
            FEATURE_KEYS.advantage: n_step_advantage.transpose(),
            FEATURE_KEYS.value_target: (
                n_step_advantage + mb_values[:, :-1]
            ).transpose(),
        }

        full_input.update(self.action_processor.combine_batch(mb_actions))
        full_input.update(self.obs_processor.combine_batch(mb_obs))
        full_input = {k: combine_first_dimensions(v) for k, v in full_input.items()}

        if not self.do_training:
            pass
        else:
            self.agent.train(full_input)

        self.latest_obs = latest_obs
        self.batch_counter += 1
        #print(f"Number of total game steps: {self.agent.get_training_step()}")

        sys.stdout.flush()
        return True
