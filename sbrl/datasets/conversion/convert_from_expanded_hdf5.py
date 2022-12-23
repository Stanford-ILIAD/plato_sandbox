from argparse import ArgumentParser

import h5py
import numpy as np

from sbrl.envs.param_spec import ParamEnvSpec
from sbrl.envs.robosuite.robosuite_env import get_rs_example_spec_params

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('file', type=str, help='hdf5 file to load from')
    parser.add_argument('env_name', type=str, help='what type of environment is this? [NutAssemblySquare, ToolHang] supported so far')
    # parser.add_argument('--save_file', type=str, default=None, help='Optional npz file to output to')
    parser.add_argument('--mask', type=str, default=None, help='Optional episode mask (will look under mask/{}) by default')
    parser.add_argument('--add_obs_keys', type=str, nargs='+', required=True,
                        help='These keys will be concatenated into key obs/object')
    parser.add_argument('--add_ee_ori', action='store_true')
    parser.add_argument('--imgs', action='store_true')
    args = parser.parse_args()

    # loading robosuite data
    file = args.file  # "data/robosuite/human_square_low_dim.hdf5", for example
    env_name = args.env_name  # "NutAssemblySquare", for example
    # file = "human_tool_hang_low_dim.hdf5"

    es_prms = get_rs_example_spec_params(env_name, raw=True)
    if args.imgs:
        es_prms.observation_names.append('image')
    env_spec = ParamEnvSpec(es_prms)

    obs_keys = list(args.add_obs_keys)

    with h5py.File(file, 'r+') as node:
        # will load
        num_demos = len(node['data'].keys())
        for i in range(1, num_demos+1):
            ep = node['data'][f'demo_{i}']
            obs = ep['obs']
            assert all(k in obs.keys() for k in obs_keys), [list(obs.keys()), obs_keys]
            if 'object' not in obs.keys():
                all_arr = np.concatenate([obs[k] for k in obs_keys], axis=-1)
                obs['object'] = all_arr

            if 'robot0_eef_vel_lin' not in obs.keys():
                obs['robot0_eef_vel_lin'] = np.zeros_like(obs['robot0_eef_pos'])
                obs['robot0_eef_vel_ang'] = np.zeros_like(obs['robot0_eef_pos'])

            if 'robot0_joint_pos' not in obs.keys():
                obs['robot0_joint_pos'] = obs['robot0_proprio-state'][..., :7]

    # if args.add_ee_ori:
    #     new_data = d()
    #     for key in data.leaf_keys():
    #         new_data[key] = data[key]
    #         if "_eef_quat" in key:
    #             # euler
    #             new_data[key.replace("_eef_quat", "_eef_eul")] = Rotation.from_quat(data[key]).as_euler("xyz")
    #     data = new_data
    #
    # logger.debug(f"Done loading. Dataset length = {len(data['done'])}")
    #
    # if args.save_file is None:
    #     data.leaf_shapes().pprint()
    # else:
    #     logger.warn(f"Saving to --> {args.save_file}")
    #     np.savez_compressed(args.save_file, **data.as_dict())