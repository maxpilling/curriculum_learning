"""multienv

Functions and classes for working with multiple SC2 environments.
"""

from multiprocessing import Process, Pipe
from pysc2.env import sc2_env


# The following (worker, CloudpickleWrapper, SubprocVecEnv) is copied from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
# with some SCII specific modifications taken from
# https://github.com/pekaalto/sc2aibot


def worker(remote, env_fn_wrapper):
    """worker

    Handling the:
    action -> [action] and  [timestep] -> timestep
    single-player conversions here

    :param remote: The remote object to receive actions from.
    :param env_fn_wrapper: The environment function wrapper.
    """

    # Get the function wrappers current value.
    env = env_fn_wrapper.current_input()

    # Given a specific input, deal with the
    # command and either increment, reset or
    # fully close the remote.
    while True:
        cmd, action = remote.recv()
        if cmd == "step":
            timesteps = env.step([action])
            assert len(timesteps) == 1
            remote.send(timesteps[0])
        elif cmd == "reset":
            timesteps = env.reset()
            assert len(timesteps) == 1
            remote.send(timesteps[0])
        elif cmd == "close":
            remote.close()
            break
        else:
            raise NotImplementedError


class CloudpickleWrapper:
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, current_input):
        self.current_input = current_input

    def __getstate__(self):
        import cloudpickle

        return cloudpickle.dumps(self.current_input)

    def __setstate__(self, observation):
        import pickle

        self.current_input = pickle.loads(observation)


class SubprocVecEnv:
    """SubprocVecEnv

    Links with the worker class to keep track of the timesteps and given
    commands.
    """

    def __init__(self, env_fns):
        n_envs = len(env_fns)
        self.game_steps = 0
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(n_envs)])

        self.processes = [
            Process(target=worker, args=(work_remote, CloudpickleWrapper(env_fn)))
            for (work_remote, env_fn) in zip(self.work_remotes, env_fns)
        ]

        for process in self.processes:
            process.start()

        self.n_envs = n_envs
    def get_steps(self):
        return self.game_steps

    def _step_or_reset(self, command, actions=None):
        actions = actions or [None] * self.n_envs

        self.game_steps = self.game_steps + 1
        for remote, action in zip(self.remotes, actions):
            remote.send((command, action))

        timesteps = [remote.recv() for remote in self.remotes]

        return timesteps

    def step(self, actions):
        """step

        Take one step of the environment.
        """

        return self._step_or_reset("step", actions)

    def reset(self):
        """reset

        Reset the environment.
        """

        return self._step_or_reset("reset", None)

    def close(self):
        """reset

        Close the envrionment.
        """

        for remote in self.remotes:
            remote.send(("close", None))

        for process in self.processes:
            process.join()

    def reset_done_envs(self):
        """reset_done_envs

        Reset complete envrionments.
        """
        pass


def make_sc2env(**kwargs):
    """make_sc2env

    Wrap the SCII environment command, to setup a SCII
    environment given a set of arguments.

    :param **kwargs: The supplied arguments.
    """

    env = sc2_env.SC2Env(**kwargs)
    return env
