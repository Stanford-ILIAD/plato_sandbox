#!/usr/bin/env python3
import sys

import numpy as np

sys.path.append('./Eval')
sys.path.append('/')

from env_40 import Engine40


class Engine42(Engine40):
    def __init__(self, worker_id, opti, p_id, taskId=5, maxSteps=15, n_dmps=3, cReward=True):
        super(Engine42,self).__init__(worker_id, opti, p_id, taskId=taskId, maxSteps=maxSteps, n_dmps=n_dmps, cReward=cReward)
        self.opti = opti

    def get_success(self,seg=None):
        cur_obj = np.array(self.p.getBasePositionAndOrientation(self.obj_id)[0])
        dist = np.linalg.norm(cur_obj - self.box_pos)
        if dist < self.dist - 0.05:
          return True
        else:
          return False

