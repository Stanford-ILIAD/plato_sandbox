#!/usr/bin/env python3
import sys

import numpy as np

sys.path.append('./Eval')
sys.path.append('/')

from env_43 import Engine43


class Engine45(Engine43):
    def __init__(self, worker_id, opti, p_id, taskId=5, maxSteps=15, n_dmps=3, cReward=True):
        super(Engine45,self).__init__(worker_id, opti, p_id, taskId=taskId, maxSteps=maxSteps, n_dmps=n_dmps, cReward=cReward)
        self.opti = opti

    def get_success(self,seg=None):
        obj_pos = np.array(self.p.getBasePositionAndOrientation(self.obj_id)[0])
        if obj_pos[2] > self.height + 0.05:
          return True
        else:
          return False
                          
