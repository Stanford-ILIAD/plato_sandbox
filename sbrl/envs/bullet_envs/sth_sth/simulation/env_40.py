#!/usr/bin/env python3
import math
import os
import random
import sys

import numpy as np

sys.path.append('/')

from env import Engine


class Engine40(Engine):
    def __init__(self, worker_id, opti, p_id, taskId=5, maxSteps=15, n_dmps=3, cReward=True):
        super(Engine40,self).__init__(opti, wid=worker_id, p_id=p_id, maxSteps=maxSteps, taskId=taskId, n_dmps=n_dmps, cReward=cReward,robot_model=None)
        self.opti = opti

    def init_obj(self):
        self.obj_file = os.path.join(self.urdf_dir,"objmodels/nut.urdf")
        self.obj_position = np.array([0.3637 + 0.06, -0.06, 0.35])
        self.obj_scaling = 2
        self.obj_orientation = self.p.getQuaternionFromEuler([math.pi/2+0.2, -math.pi/2, -0.3])
        self.obj_id = self.p.loadURDF(fileName=self.obj_file, basePosition=self.obj_position,baseOrientation=self.obj_orientation,
                                     globalScaling=self.obj_scaling)#,physicsClientId=self.physical_id

        texture_path = os.path.join(self.resources_dir,'textures/sun_textures')
        texture_file = os.path.join(texture_path,random.sample(os.listdir(texture_path),1)[0])
        textid = self.p.loadTexture(texture_file)
#        self.p.changeVisualShape (self.obj_id, -1, textureUniqueId=textid)
        self.p.changeVisualShape (self.obj_id, -1, rgbaColor=[1,1,1,1])


        self.box_file = os.path.join(self.urdf_dir,"objmodels/urdfs/cup.urdf")
        self.box_position = np.array([0.34, 0.16, 0.33])
        self.box_orientation = self.p.getQuaternionFromEuler([-math.pi/2, 0, 0])
        self.box_scaling = 0.15
        self.box_id = self.p.loadURDF(fileName=self.box_file, basePosition=self.box_position,baseOrientation=self.box_orientation,
                                     globalScaling=self.box_scaling,useFixedBase=True)
        self.p.changeVisualShape (self.box_id, -1, rgbaColor=[1, 0, 0, 1])
        self.dist = 0.0

    def reset_obj(self):
        self.p.resetBasePositionAndOrientation(self.obj_id,self.obj_position,self.obj_orientation)
        box_diff = np.random.uniform(-0.1,0.1,size=(2,))
        self.box_pos = np.copy(self.box_position)
        self.box_pos[0] += box_diff[0]
        self.box_pos[1] += box_diff[1]
        self.p.resetBasePositionAndOrientation(self.box_id,self.box_pos,self.box_orientation) 

    def init_motion(self):
        self.data_q = np.load (os.path.join(self.robot_recordings_dir,"47-4/q.npy"))
        self.data_gripper = np.load (self.configs_dir + '/init/gripper.npy')
        self.robot.setJointValue(self.data_q[0],gripper=self.data_gripper[0])

    def init_grasp(self):
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
        pos[0] += -.1
        orn = self.robot.getEndEffectorOrn()
        for i in range(109):
           self.robot.operationSpacePositionControl(pos,orn,null_pose=self.null_q,gripperPos=130)

        cur_joint = self.robot.get_joint_values()
        cur_pos = self.robot.getEndEffectorPos()
        cur_orn = self.robot.getEndEffectorOrn()
        pos_diff = np.random.uniform(-0.1,0.1,size=(2,))
        cur_pos[:2] = cur_pos[:2] + pos_diff
        cur_pos[2] += 0.15
        for i in range(19):
           self.robot.operationSpacePositionControl(cur_pos,cur_orn,null_pose=cur_joint,gripperPos=130)
        self.start_pos = self.p.getLinkState (self.robotId, 7)[0]
        
        cur_obj = np.array(self.p.getBasePositionAndOrientation(self.obj_id)[0])
        self.dist = np.linalg.norm(cur_obj - self.box_pos)       
 
    def get_success(self,seg=None):
        cur_obj = np.array(self.p.getBasePositionAndOrientation(self.obj_id)[0])
        dist = np.linalg.norm(cur_obj - self.box_pos)       
        if dist > self.dist + 0.05:
          return True
        else:
          return False
