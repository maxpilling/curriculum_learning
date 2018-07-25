import numpy as np

from collections import namedtuple
from pysc2.env.environment import TimeStep, StepType
from pysc2.lib import actions
from pysc2.lib.features import SCREEN_FEATURES, MINIMAP_FEATURES, FeatureType

# All code taken from https://github.com/pekaalto/sc2aibot and then tidied
# with comments and syntax changes.


def log_transform(x, scale):
    """log_transform

    Given an input x, scale it and take the logarithm.

    :param x: The input variable.
    :param scale: How much to scale the value by.
    """

    # 8 is a "feel good" magic number and doesn't mean anything here.
    return np.log(8 * x / scale + 1)


def get_visibility_flag(visibility_feature):
    """get_visibility_flag

    Get the visibility of the current feature.

    :param visibility_feature: The current visibility feature.
    """

    # 0 = Hidden
    # 1 = Fogged
    # 2 = Visible
    return np.expand_dims(visibility_feature == 2, axis=0)


def numeric_idx_and_scale(input_set):
    """numeric_idx_and_scale

    Given a set, return an ID for each scalar value.

    :param input_set: The input set.
    """

    # Get the scalar values, before zipping the id and value up.
    idx_and_scale = [
        (k.index, k.scale) for k in input_set
        if k.type == FeatureType.SCALAR
    ]

    idx, scale = [np.array(k) for k in zip(*idx_and_scale)]
    scale = scale.reshape(-1, 1, 1)

    return idx, scale


def stack_list_of_dicts(d):
    """stack_list_of_dicts

    Given an input dictionary, stack the values
    into a list.

    :param d: Input dictionary.
    """

    return {key: np.stack([a[key] for a in d]) for key in d[0]}


def get_available_actions_flags(obs):
    """get_available_actions_flags

    Get the current available action flags from the SCII
    Observation object.

    :param obs: The StarCraft II Observation object.
    """

    # Return only the available actions from the Observation object.
    available_actions_dense = np.zeros(len(actions.FUNCTIONS), dtype=np.float32)
    available_actions_dense[obs['available_actions']] = 1

    return available_actions_dense


class ObsProcessor:
    """ObsProcessor

    A class dedicated to processing and making interfacing with the
    StarCraft II obs object easier.
    """

    N_SCREEN_CHANNELS = 13
    N_MINIMAP_CHANNELS = 5
    N_NON_SPATIAL = 11

    def __init__(self):
        # Define the screen and minimaps scale and ids.
        self.screen_numeric_idx, self.screen_numeric_scale = \
            numeric_idx_and_scale(SCREEN_FEATURES)

        self.minimap_numeric_idx, self.minimap_numeric_scale = \
            numeric_idx_and_scale(MINIMAP_FEATURES)

        # Finds the ids of the flags we care about in both the screen
        # and the minimap. This differs across the two, due to
        # different information being offered in both.
        screen_flag_names = ["creep", "power", "selected"]

        self.screen_flag_idx = [k.index for k in SCREEN_FEATURES
                                if k.name in screen_flag_names]

        minimap_flag_names = ["creep", "camera", "selected"]

        self.minimap_flag_idx = [k.index for k in MINIMAP_FEATURES
                                 if k.name in minimap_flag_names]

    def get_screen_numeric(self, obs):
        """get_screen_numeric

        Get and scale the screen portion of the obs object,
        and encode the visibility flags alongside it.

        :param obs: The StarCraft II Observation object.
        """

        screen_obs = obs["screen"]

        scaled_scalar_obs = log_transform(
            screen_obs[self.screen_numeric_idx], self.screen_numeric_scale
        )

        return np.r_[
            scaled_scalar_obs,
            screen_obs[self.screen_flag_idx],
            get_visibility_flag(screen_obs[SCREEN_FEATURES.visibility_map.index])
        ]

    def get_minimap_numeric(self, obs):
        """get_minimap_numeric

        Get and scale the mini-map portion of the obs object,
        and encode the visibility flags alongside it.

        :param obs: The StarCraft II Observation object.
        """

        minimap_obs = obs["minimap"]

        # This is only height_map for mini-map.
        scaled_scalar_obs = log_transform(
            minimap_obs[self.minimap_numeric_idx], self.minimap_numeric_scale
        )

        return np.r_[
            scaled_scalar_obs,
            minimap_obs[self.minimap_flag_idx],
            get_visibility_flag(minimap_obs[MINIMAP_FEATURES.visibility_map.index])
        ]

    def process_one_input(self, timestep: TimeStep):
        """process_one_input

        Process a single input to the environment by returning
        all the current needed variables in a dictionary.

        :param timestep: The current timestep to deal with.
        :type timestep: TimeStep
        """

        obs = timestep.observation

        pp_obs = {
            FEATURE_KEYS.screen_numeric: self.get_screen_numeric(obs),
            FEATURE_KEYS.screen_unit_type: obs["screen"][SCREEN_FEATURES.unit_type.index],
            FEATURE_KEYS.minimap_numeric: self.get_minimap_numeric(obs),
            FEATURE_KEYS.non_spatial_features: obs['player'],
            FEATURE_KEYS.available_action_ids: get_available_actions_flags(obs),
            FEATURE_KEYS.player_relative_screen: obs["screen"][
                SCREEN_FEATURES.player_relative.index],
            FEATURE_KEYS.player_relative_minimap: obs["minimap"][
                MINIMAP_FEATURES.player_relative.index]
        }

        return pp_obs

    def process(self, obs_list):
        """process

        Given a list of time-steps, process the input for each,
        and then return back the available actions and screen
        and mini-maps for all time-steps.

        :param obs_list: list[TimeStep],
        """

        pp_obs = [self.process_one_input(obs) for obs in obs_list]
        pp_obs = stack_list_of_dicts(pp_obs)

        for k in ["screen_numeric", "minimap_numeric"]:
            pp_obs[k] = np.transpose(pp_obs[k], [0, 2, 3, 1])

        return pp_obs

    def combine_batch(self, mb_obs):
        """combine_batch

        Combine a list of obs dictionaries.

        :param mb_obs: A list of Observation dictionaries.
        """

        return stack_list_of_dicts(mb_obs)


def make_default_args(arg_names):
    """make_default_args

    Make a default set of arguments.

    :param arg_names: A list of argument names.
    """

    default_args = []
    spatial_seen = False
    spatial_arguments = ["screen", "minimap", "screen2"]

    for k in arg_names:
        if k in spatial_arguments:
            spatial_seen = True
            continue
        else:
            assert not spatial_seen, "Got %s argument after spatial argument" % k
            default_args.append([0])

    return tuple(default_args), spatial_seen


def convert_point_to_rectangle(point, delta, dim):
    """convert_point_to_rectangle

    Convert a given point to a rectangle, defined by
    two diagonal points, bounding the rectangle.

    :param point: A given point.
    :param delta: The difference between the point and the point.
    :param dim: The size of the rectangle.
    """

    def l(x):
        return max(0, min(x, dim - 1))

    p1 = [l(k - delta) for k in point]
    p2 = [l(k + delta) for k in point]
    return p1, p2


def arg_names():
    """arg_names

    Return a list of all argument names from the actions list.
    """

    x = [[a.name for a in k.args] for k in actions.FUNCTIONS]
    assert all("minimap2" not in k for k in x)
    return x


def find_rect_function_id():
    """find_rect_function_id

    Fine the id of the rectangle function ID, used for selecting units.
    """

    x = [k.id for k, names in zip(actions.FUNCTIONS, arg_names()) if "screen2" in names]
    assert len(x) == 1
    return x[0]


class ActionProcessor:
    """ActionProcessor

    Process a given action.
    """

    def __init__(self, dim, rect_delta=5):
        self.default_args, is_spatial = zip(*[make_default_args(k) for k in arg_names()])
        self.is_spatial = np.array(is_spatial)
        self.rect_select_action_id = find_rect_function_id()
        self.rect_delta = rect_delta
        self.dim = dim

    def make_one_action(self, action_id, spatial_coordinates):
        """make_one_action

        :param action_id: The action id to perform.
        :param spatial_coordinates: The co-ordinates to perform the action at.
        """

        args = list(self.default_args[action_id])
        assert all(s < self.dim for s in spatial_coordinates)

        # If the action is a select action, then convert the point first, before performing it.
        if action_id == self.rect_select_action_id:
            args.extend(convert_point_to_rectangle(spatial_coordinates, self.rect_delta, self.dim))
        elif self.is_spatial[action_id]:
            # Flip the co-ordinates from (x, y) to (y, x).
            args.append(spatial_coordinates[::-1])

        return actions.FunctionCall(action_id, args)

    def process(self, action_ids, spatial_action_2ds):
        """process

        Process a list of actions and co-ordinates to perform them at.

        :param action_ids: The list of action IDs.
        :param spatial_action_2ds: The co-ordinates to perform them all at.
        """

        return [self.make_one_action(a_id, coord)
                for a_id, coord in zip(action_ids, spatial_action_2ds)]

    def combine_batch(self, mb_actions):
        """combine_batch

        Combine a batch of actions.

        :param mb_actions: A list of actions and co-ordinates.
        """

        d = {}
        d[FEATURE_KEYS.selected_action_id] = np.stack(k[0] for k in mb_actions)
        d[FEATURE_KEYS.selected_spatial_action] = np.stack(k[1] for k in mb_actions)

        d[FEATURE_KEYS.is_spatial_action_available] = self.is_spatial[
            d[FEATURE_KEYS.selected_action_id]
        ]

        return d


FEATURE_LIST = (
    "minimap_numeric",
    "screen_numeric",
    "non_spatial_features",
    "screen_unit_type",
    "is_spatial_action_available",
    "selected_spatial_action",
    "selected_action_id",
    "available_action_ids",
    "value_target",
    "advantage",
    "player_relative_screen",
    "player_relative_minimap"
)

AgentInputTuple = namedtuple("AgentInputTuple", FEATURE_LIST)
FEATURE_KEYS = AgentInputTuple(*FEATURE_LIST)
