#!/usr/bin/env python3

import sys
sys.path.append('./Eval')
sys.path.append('/')

from env_121 import Engine121


class Engine171(Engine121):
    def __init__(self, worker_id, opti, p_id, taskId=5, maxSteps=15, n_dmps=3, cReward=True):
        super(Engine171,self).__init__(worker_id, opti, p_id, taskId=taskId, maxSteps=maxSteps, n_dmps=n_dmps, cReward=cReward)
        self.opti = opti


