import os
from argparse import ArgumentParser

from sbrl.datasets.fast_np_interaction_dataset import NpInteractionDataset
from sbrl.utils.config_utils import get_path_module
from sbrl.utils.python_utils import AttrDict as d


np_dataset_module = get_path_module("np_dataset_module", 
                                    os.path.join(os.path.dirname(__file__), "np_dataset_config.py"))


def declare_arguments(parser=ArgumentParser()):
    return (np_dataset_module.params >> "declare_arguments")(parser)


# strictly ordered processing order
def process_params(group_name, common_params):
    common_params = (np_dataset_module.params >> "process_params")(group_name, common_params)
    utils = common_params >> 'utils'
    sampling_prms = common_params << "model/params/sampling_params"
    if sampling_prms is None:
        sampling_prms = d(
            sample_goals=True,  # we always want to get the goal.
            sample_pre_window=True,
            sample_goal_start=False,
            sample_post_goals_only=False,
            sample_interaction_goals_only=False,
            pre_window_key_prefix="initiation",
            goal_key_prefix="goal_states",
            soft_boundary_length=common_params >> 'horizon',
            init_soft_boundary_length=common_params >> 'horizon',
        )

    # faster data loading scheme for contact.
    common_params[group_name].cls = NpInteractionDataset
    common_params[group_name]\
        .params.parse_interaction_bounds_from_episode_fn = utils.parse_interaction_bounds_from_episode_fn
    common_params[group_name].params.combine(sampling_prms)
    
    return common_params


params = d(
    declare_arguments=declare_arguments,
    process_params=process_params,
)
