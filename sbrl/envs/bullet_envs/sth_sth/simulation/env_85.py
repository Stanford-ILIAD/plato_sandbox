#!/usr/bin/env python3
import math
import os
import sys

import numpy as np

sys.path.append('/')

from env import Engine


class Engine85(Engine):
    def __init__(self, worker_id, opti, p_id, taskId=5, maxSteps=15, n_dmps=3, cReward=True):
        super(Engine85,self).__init__(opti, wid=worker_id, p_id=p_id, maxSteps=maxSteps, taskId=taskId, n_dmps=n_dmps, cReward=cReward,robot_model=None)
        self.opti = opti

    def init_obj(self):
        self.box_file = os.path.join (self.urdf_dir, "openbox2/openbox.urdf")
        print("self.box_file",self.box_file)
        self.box_position = [0.40, 0.02, 0.34]
        self.box_scaling = 0.00035
        self.box_orientation = self.p.getQuaternionFromEuler ([0, 0, math.pi/2])
        self.box_id = self.p.loadURDF (fileName=self.box_file, basePosition=self.box_position,
                                      baseOrientation=self.box_orientation,
                                      globalScaling=self.box_scaling,useFixedBase=True)

        self.obj_file = os.path.join(self.urdf_dir,"objmodels/nut.urdf")
        self.obj_position = [0.3637 + 0.06, -0.06, 0.35]
        self.obj_scaling = 2
        self.obj_orientation = self.p.getQuaternionFromEuler([math.pi/2+0.2, -math.pi/2, -0.3])
        self.obj_id = self.p.loadURDF(fileName=self.obj_file, basePosition=self.obj_position,baseOrientation=self.obj_orientation,
                                     globalScaling=self.obj_scaling)#,physicsClientId=self.physical_id
        self.pos = None

        
    def reset_obj(self):
       self.p.resetBasePositionAndOrientation(self.obj_id,self.obj_position,self.obj_orientation)


    def init_motion(self):
        self.data_q = np.load (os.path.join(self.robot_recordings_dir,"47-4/q.npy"))
        self.data_gripper = np.load (self.configs_dir + '/init/gripper.npy')
        self.robot.setJointValue(self.data_q[0],gripper=self.data_gripper[0])

    def init_motion(self):
        self.data_q = np.load (os.path.join(self.robot_recordings_dir,"47-4/q.npy"))
        self.data_gripper = np.load (self.configs_dir + '/init/gripper.npy')
        self.robot.setJointValue(self.data_q[0],gripper=self.data_gripper[0])

    def init_grasp(self):
        self.box_position[2] = -.34
        self.p.resetBasePositionAndOrientation(self.box_id,self.box_position,self.box_orientation)

        self.robot.gripper_control(0)

        qlist = np.load( os.path.join(self.robot_recordings_dir, "47-4/q.npy"))
        glist = np.load( os.path.join(self.robot_recordings_dir, "47-4/gripper.npy"))
        num_q = len(qlist[0])
        self.fix_orn = np.load (os.path.join (self.configs_dir, 'init', 'orn.npy'))
        self.null_q = qlist[180]
        self.robot.setJointValue(qlist[40],glist[40])
        for i in range(40,180,1):
            glist[i] = min(130,glist[i])
            self.robot.jointPositionControl(qlist[i],gripper=glist[i])

        pos = self.robot.getEndEffectorPos()
        pos[1] += 0.1
        pos[0] += 0.11
        pos[2] += 0.03
        orn = self.robot.getEndEffectorOrn()
        for i in range(109):
           self.robot.operationSpacePositionControl(pos,orn,null_pose=self.null_q,gripperPos=130)
#        time.sleep(3)
        self.start_pos = self.p.getLinkState (self.robotId, 7)[0]
        self.box_position[2] *= -1.0
        self.p.resetBasePositionAndOrientation(self.box_id,self.box_position,self.box_orientation)
        self.pos = self.p.getBasePositionAndOrientation(self.obj_id)[0]
 
    def get_success(self,seg=None):
        obj_pos = self.p.getBasePositionAndOrientation(self.obj_id)[0]
        # check whether the object is still in the gripper
        left_closet_info = self.p.getContactPoints (self.robotId, self.obj_id, self.robot.gripper_left_tip_index, -1)
        right_closet_info = self.p.getContactPoints (self.robotId, self.obj_id, self.robot.gripper_right_tip_index, -1)
        if len (left_closet_info) > 0 and len (right_closet_info) > 0 and obj_pos[1] < self.pos[1] - 0.05:
          return True
        else:
          return False

