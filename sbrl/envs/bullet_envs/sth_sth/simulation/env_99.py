#!/usr/bin/env python3

import sys
sys.path.append('/')

from env_98 import Engine98


class Engine99(Engine98):
    def __init__(self, worker_id, opti, p_id, taskId=5, maxSteps=15, n_dmps=3, cReward=True):
        super(Engine99,self).__init__(worker_id, opti, p_id, taskId=taskId, maxSteps=maxSteps, n_dmps=n_dmps, cReward=cReward)
        self.opti = opti

    def get_success(self,seg=None):
        pos = self.p.getBasePositionAndOrientation(self.obj_id)[0][2]
        closet_info = self.p.getContactPoints (self.robotId, self.obj_id)
        if len(closet_info) > 0:
          self.contact = True
        if pos < self.pos - 0.04 and self.dmp.timestep >= self.dmp.timesteps and self.contact:
          return True
        else:
          return False
