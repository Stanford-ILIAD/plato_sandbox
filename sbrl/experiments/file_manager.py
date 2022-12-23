"""
A centralized way of dealing with saving/loading files
"""

import os
import shutil
import subprocess

from sbrl.experiments import logger


class FileManager(object):
    base_dir = os.path.abspath(__file__)[:os.path.abspath(__file__).find('sbrl/')]
    configs_dir = os.path.join(base_dir, 'configs')
    data_dir = os.path.join(base_dir, 'data')
    plot_dir = os.path.join(base_dir, 'plots')
    video_dir = os.path.join(base_dir, 'videos')
    logs_dir = os.path.join(base_dir, 'logs')  # sbatch

    def __init__(self):
        pass


class ExperimentFileManager(FileManager):

    experiments_dir = os.path.join(FileManager.base_dir, 'experiments')

    def __init__(self, exp_name, is_continue=False, config_fname=None, log_fname=None, extra_args=None):
        from sbrl.utils.config_utils import get_config_args
        super(ExperimentFileManager, self).__init__()
        self._exp_name = exp_name
        self._exp_dir = os.path.join(ExperimentFileManager.experiments_dir, self._exp_name)

        # NOTE: This is very important to have. It helps prevent you from accidentally overwriting previous experiments!
        if is_continue:
            assert os.path.exists(self._exp_dir),\
                'Experiment folder "{0}" does not exists, but continue = True'.format(self._exp_name)
        else:
            assert not os.path.exists(self._exp_dir),\
                'Experiment folder "{0}" exists, but continue = False'.format(self._exp_name)

        # NOTE: This is very important to have. If you save the current git commit and diff, you can always reproduce.
        self._save_git()

        # NOTE: This is very important to have. It lets you quickly examine the configuration of old experiments.
        if config_fname is not None:
            shutil.copy(config_fname, os.path.join(self._exp_dir, 'config.py'))
            # write all command line args (for config) as well, if they are provided
            if extra_args is None or len(extra_args) == 0:
                c_args = get_config_args()
            else:
                c_args = list(extra_args)

            if len(c_args) > 0:
                arg_string = ""
                for c in c_args:
                    arg_string += c + " "
                with open(os.path.join(self._exp_dir, 'config_args.txt'), "w") as f:
                    f.write(arg_string)

        if log_fname is not None:
            logger.setup(os.path.join(self._exp_dir, log_fname))

    def _save_git(self):
        git_dir = os.path.join(self._exp_dir, 'git')
        os.makedirs(git_dir, exist_ok=True)

        git_commit_fname = os.path.join(git_dir, 'commit.txt')
        git_diff_fname = os.path.join(git_dir, 'diff.txt')

        if not os.path.exists(git_commit_fname):
            subprocess.call('cd {0}; git log -1 > {1}'.format(git_dir, git_commit_fname), shell=True)
        if not os.path.exists(git_diff_fname):
            subprocess.call('cd {0}; git diff > {1}'.format(git_dir, git_diff_fname), shell=True)

    ###############
    # Experiments #
    ###############

    @property
    def exp_dir(self):
        os.makedirs(self._exp_dir, exist_ok=True)
        return self._exp_dir

    ##########
    # Models #
    ##########

    @property
    def models_dir(self):
        models_dir = os.path.join(self._exp_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        return models_dir

    @property
    def exp_video_dir(self):
        videos_dir = os.path.join(self.video_dir, self._exp_name)
        os.makedirs(videos_dir, exist_ok=True)
        return videos_dir

    # TODO: add more as appropriate
