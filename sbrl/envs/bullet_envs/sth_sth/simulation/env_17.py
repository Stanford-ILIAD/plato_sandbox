#!/usr/bin/env python3
import sys
from math import sin, cos, acos

import numpy as np

sys.path.append('/')

from env_16 import Engine16


def angleaxis2quaternion(angleaxis):
  angle = np.linalg.norm(angleaxis)
  axis = angleaxis / (angle + 0.00001)
  q0 = cos(angle/2)
  qx,qy,qz = axis * sin(angle/2)
  return np.array([qx,qy,qz,q0])

def quaternion2angleaxis(quater):
  angle = 2 * acos(quater[3])
  axis = quater[:3]/(sin(angle/2)+0.00001)
  angleaxis = axis * angle
  return np.array(angleaxis)


class Engine17(Engine16):
    def __init__(self, worker_id, opti, p_id, taskId=5, maxSteps=15, n_dmps=3, cReward=True):
        super(Engine17,self).__init__(worker_id, opti, p_id, taskId=taskId, maxSteps=maxSteps, n_dmps=n_dmps, cReward=cReward)
        self.opti = opti
    
    def get_success(self,seg=None):
        box = self.p.get_aabb(self.box_id)
        box_center = [(x + y) * 0.5 for x, y in zip (box[0], box[1])]
        obj = self.p.get_aabb(self.obj_id)
        obj_center = [(x + y) * 0.5 for x, y in zip (obj[0], obj[1])]

        # check whether the object is still in the gripper
        left_closet_info = self.p.getContactPoints (self.robotId, self.obj_id, self.robot.gripper_left_tip_index, -1)
        right_closet_info = self.p.getContactPoints (self.robotId, self.obj_id, self.robot.gripper_right_tip_index, -1)
        #print(len (left_closet_info),len (right_closet_info),obj[0][0], box[1][0])
        if len (left_closet_info) > 0 and len (right_closet_info) > 0 and obj[0][0] > box[1][0]:
          return True
        else:
          return False
