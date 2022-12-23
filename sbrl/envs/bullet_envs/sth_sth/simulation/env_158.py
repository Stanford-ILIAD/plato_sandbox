#!/usr/bin/env python3

import sys
sys.path.append('./Eval')
sys.path.append('/')

from env_57 import Engine57


class Engine158(Engine57):
    def __init__(self, worker_id, opti, p_id, taskId=5, maxSteps=15, n_dmps=3, cReward=True):
        super(Engine158,self).__init__(worker_id, opti, p_id, taskId=taskId, maxSteps=maxSteps, n_dmps=n_dmps, cReward=cReward)
        self.opti = opti

