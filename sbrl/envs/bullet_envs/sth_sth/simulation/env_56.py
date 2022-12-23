#!/usr/bin/env python3
import sys

import numpy as np

sys.path.append('./Eval')
sys.path.append('/')

from env_55 import Engine55


class Engine56(Engine55):
    def __init__(self, worker_id, opti, p_id, taskId=5, maxSteps=15, n_dmps=3, cReward=True):
        super(Engine56,self).__init__(worker_id, opti, p_id, taskId=taskId, maxSteps=maxSteps, n_dmps=n_dmps, cReward=cReward)
        self.opti = opti

    def get_success(self,seg=None):
        pos = np.array(self.p.getBasePositionAndOrientation(self.obj_id)[0])
        dist = np.linalg.norm(pos-self.pos)
        contact_info = self.p.getContactPoints (self.robotId, self.obj_id)
        if len(contact_info) > 0:
          self.contact = True
        if dist < 0.05 and self.contact and self.dmp.timestep >= self.dmp.timesteps:
          return True
        else:
          return False
