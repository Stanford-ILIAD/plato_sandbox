"""
Evals multi-task play environment with a scripted policy
"""

import numpy as np
import torch

from sbrl.experiments import logger
from sbrl.experiments.grouped_parser import GroupedArgumentParser
from sbrl.utils.python_utils import timeit, exit_on_ctrl_c, AttrDict
from sbrl.utils.script_utils import eval_preamble
from sbrl.utils.torch_utils import expand_h, to_numpy

exit_on_ctrl_c()

# things we can use from command line
macros = {}

grouped_parser = GroupedArgumentParser(macros=macros)
grouped_parser.add_argument('config', type=str)
grouped_parser.add_argument('--model_file', type=str)
grouped_parser.add_argument('--num_eps', type=int, required=True)
grouped_parser.add_argument('--num_proc', type=int, default=1)
grouped_parser.add_argument('--no_model_file', action="store_true")
grouped_parser.add_argument('--random_policy', action="store_true")
local_args, remaining_args = grouped_parser.parse_local_args()

# policy generates the goals
ordered_modules = ['env_spec', 'env_train', 'model', 'dataset_train', 'policy', 'policy_hardcoded']

exp_name, args, params, model_fname = eval_preamble(local_args.config, remaining_args, grouped_parser, ordered_modules, local_args.model_file, debug=False)

# create all the global modules
env_spec = params.env_spec.cls(params.env_spec.params)
model = params.model.cls(params.model.params, env_spec, None)

# utils required to know the success metrics
utils = params >> 'utils'
# lookup from policy_name --> success_metric_fn (batch-supported)
success_metric_fns: dict = utils.get_success_metric_fns()

# restore model
if not args.no_model_file:
    model.restore_from_file(model_fname)


def eval_rollout(local_args, prefix, env, obs, goal, policy, policy_hardcoded, stack_goal=True):
    obs_names = obs.list_leaf_keys()
    goal_obs, goal_goal = obs, goal
    ptype, raw_name, name = -1, None, None
    # stop counter for policy rollout
    counter = 0

    # helpers for detecting & stopping on success
    def prep_for_success_fn(obs, goal):
        return (obs, np.array([0]), goal)

    def stop_condition(obs, goal, name):
        nonlocal counter
        tobs, didx, gl = prep_for_success_fn(obs, goal)
        if raw_name is None:
            this_success = False
        else:
            succ_out = success_metric_fns[raw_name](tobs, didx, gl)
            # success is range [0.5, inf]
            this_success = (succ_out >> 'best') >= 0.5
        if this_success or counter > 0:
            counter += 1

        if counter > 5:
            counter = 0
            return True

        return False
    
    # logger.debug(f'pos before: {obs.objects.position} {obs.cabinet.joint_position} {obs.drawer.joint_position}')

    # Run HARDCODED POLICY to GET GOAL
    for i in range(1000):
        with timeit("hardcoded/policy"):
            act = policy_hardcoded.get_action(model, expand_h(goal_obs), expand_h(goal_goal))

        if i == 0:
            if policy_hardcoded.curr_policy_idx == -1:
                logger.warn(f"{prefix}Policy type is -1! skipping this rollout")
                break

            ptype = policy_hardcoded.curr_policy_idx

            raw_name = policy_hardcoded.curr_policy.curr_name
            name = raw_name.replace("_", " ").title()

        with timeit("hardcoded/env_step"):
            goal_obs, goal_goal, done = env.step(act)
        
        # finish early
        if done[0] or policy_hardcoded.is_terminated(model, goal_obs, goal_goal):
            break

    # if terminated early, skip rollout
    if i == 0:
        return None, None, None, None

    # logger.debug(f'goal pos: {goal_obs.objects.position} {goal_obs.cabinet.joint_position} {goal_obs.drawer.joint_position}')
    
    # Run POLICY
    sequence = []
    
    # start from root start, make sure this reset() is properly implemented!
    presets = obs.leaf_apply(lambda arr: arr[0])
    obs, goal = env.reset(presets)
    
    # logger.debug(f'new pos: {obs.objects.position} {obs.cabinet.joint_position} {obs.drawer.joint_position}')

    # run eval for 50 extra steps than hardcoded.
    run_time = i + 50

    # nested_goal_obs = AttrDict(goal=goal_obs.leaf_apply(lambda arr: arr.copy()))
    
    for i in range(run_time):
        if stack_goal:
            # stack B x 2 x ...
            policy_obs = obs & AttrDict.leaf_combine_and_apply([obs, goal_obs], lambda vs: np.stack(vs, axis=1))
            policy_goal = AttrDict()
        else:
            # pass in as separate obs of B x 1 x ...
            policy_obs = expand_h(obs)
            policy_goal = expand_h(goal_obs)

        with timeit("proposal/policy"):
            with torch.no_grad():
                # ignore the action, sample the plan
                action = policy.get_action(model, policy_obs, policy_goal, sample_first=False)
                action.leaf_modify(lambda arr: to_numpy(arr, check=True))

        # starts from first obs & action
        sequence.append(obs & (action < env_spec.action_names))

        with timeit("proposal/env_step"):
            obs, goal, done = env.step(action)

        # optional stopping condition.
        if i > 1 and stop_condition(obs, goal_obs, raw_name):
            logger.debug(f"{prefix}Early stopping at i = {i}")
            break

    # stack arrays
    full_seq = AttrDict.leaf_combine_and_apply(sequence, lambda vs: np.stack(vs, axis=1))
    
    # logger.debug(f'final pos: {obs.objects.position} {obs.cabinet.joint_position} {obs.drawer.joint_position}')

    # compute success across all steps
    tobs, didx, gl = prep_for_success_fn((full_seq > obs_names).leaf_apply(lambda arr: arr[0]), goal_obs)
    success_dc = success_metric_fns[raw_name](tobs, didx, gl)

    # if local_args.timeit:
    #     print(timeit)
    #     timeit.reset()

    return full_seq, name, ptype, success_dc


def eval_process(inps):
    proc_id, num_eps = inps
    prefix = f"[{proc_id}]: "
    env = params.env_train.cls(params.env_train.params, env_spec)
    policy = params.policy.cls(params.policy.params, env_spec, env=env)
    policy_hardcoded = params.policy_hardcoded.cls(params.policy_hardcoded.params, env_spec, env=env)

    # begin eval loop
    successes_by_name = {}
    ep = 0
    while ep < num_eps:
        # reset environment
        obs, goal = env.reset()
        policy.reset_policy(next_obs=obs, next_goal=goal)
        policy_hardcoded.reset_policy(next_obs=obs, next_goal=goal)
        policy.warm_start(model, obs, goal)

        full_seq, pname, ptype, success_dc = eval_rollout(local_args, prefix, env, obs, goal,
                                                          policy, policy_hardcoded, stack_goal=True)
        if success_dc is not None:
            succ = (success_dc >> 'best')[0]
            logger.debug(f"{prefix}Episode {ep}: policy={pname}, success#={succ}")
            if pname not in successes_by_name.keys():
                successes_by_name[pname] = []
            successes_by_name[pname].append(int(succ >= 0.5))
            ep += 1
    
    return (successes_by_name, )


if __name__ == "__main__":
    num_proc = local_args.num_proc
    assert num_proc > 0
    logger.debug(f"Num proc: {num_proc}")
    
    # splitting the work
    assert local_args.num_eps % num_proc == 0, "Work must be evenly divisable"
    
    if num_proc > 1:
        import torch.multiprocessing as mp
        mp.set_start_method('forkserver')

        all_args = []
        for i in range(num_proc):
            all_args.append((i, local_args.num_eps // num_proc))

        logger.debug(f"launching {num_proc} processes...")
        with mp.Pool(num_proc) as p:
            rets = p.map(eval_process, all_args)
        
        # combine success data
        successes_by_name = {}
        for succ, in rets:
            for key in succ.keys():
                if key not in successes_by_name:
                    successes_by_name[key] = []
                successes_by_name[key] = successes_by_name[key] + succ[key]
        
    else:
        successes_by_name, = eval_process((0, local_args.num_eps))
    
    logger.debug("Successes:")
    AttrDict.from_dict(successes_by_name).leaf_apply(lambda vs: np.mean(vs)).pprint()

    logger.debug("Done.")
