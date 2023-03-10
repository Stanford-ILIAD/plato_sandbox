#!/usr/bin/env python3
import math
import os
import sys

import numpy as np
import pybullet as p

sys.path.append('/')

from env import Engine

#################################

np.set_printoptions(precision=4,suppress=True,linewidth=300)

class Engine93(Engine):
    def __init__(self, worker_id, opti, p_id, taskId=5, maxSteps=15, n_dmps=3, cReward=True):
        super(Engine93,self).__init__(opti, wid=worker_id, p_id=p_id, maxSteps=maxSteps, taskId=taskId, n_dmps=n_dmps, cReward=cReward,robot_model=None)
        self.opti = opti
        self.wid = worker_id
        self.p = p_id
        self.p.setPhysicsEngineParameter(enableConeFriction=1)
        self.p.setPhysicsEngineParameter(contactBreakingThreshold=0.001)
        self.p.setPhysicsEngineParameter(allowedCcdPenetration=0.0)

        self.p.setPhysicsEngineParameter(numSolverIterations=20)
        self.p.setPhysicsEngineParameter(numSubSteps=10)

        self.p.setPhysicsEngineParameter(enableFileCaching=0)

        self.p.setTimeStep(1 / 30.0)
        self.p.setGravity(0, 0, -9.81)

        expert_traj_dir = os.path.join(self.robot_recordings_dir,"87-0")
        self.data_q = np.load(os.path.join(expert_traj_dir,'q.npy'))
        self.data_dq = np.load(os.path.join(expert_traj_dir,'dq.npy'))

        pos_list = []
        orn_list = []
        for i in range(len(self.data_q)):
          self.robot.setJointValue(self.data_q[i],gripper=220)
          pos = self.robot.getEndEffectorPos()
          pos_list.append(pos)
          orn = self.robot.getEndEffectorOrn()
          orn_list.append(orn)
        self.pos_traj = pos_list
        self.orn_traj = orn_list

        self.fix_orn = orn_list

        self.count = 0

    def init_obj(self):
        self.obj_file = os.path.join(self.urdf_dir,"objmodels/nut.urdf")
        self.obj_position = [0.35, 0.1, 0.31]
        self.obj_scaling = 2
        self.obj_orientation = self.p.getQuaternionFromEuler([math.pi/2, -math.pi/2, 0])
        self.obj_id = self.p.loadURDF(fileName=self.obj_file, basePosition=self.obj_position,baseOrientation=self.obj_orientation,globalScaling=self.obj_scaling)


    def reset_obj(self):
        obj_x = 0.35
        obj_y = 0.1
        transl = np.random.uniform(-0.1,0.1,size=(2,))
        self.obj_position = [obj_x + transl[0], obj_y + transl[1], 0.31]
        self.obj_orientation = self.p.getQuaternionFromEuler ([-math.pi / 2, 0, 0])
        self.p.resetBasePositionAndOrientation (self.obj_id, self.obj_position, self.obj_orientation)
        p.changeVisualShape (self.obj_id, -1, rgbaColor=[1, 0, 0, 1])
        for i in range(5):
          self.p.stepSimulation()


    def init_grasp(self):
        for j in range (7):
            self.p.resetJointState(self.robotId, j, self.data_q[0][j], self.data_dq[0][j])
        self.robot.gripper_control(0)

        self.null_q = self.data_q[0]
        orn = self.orn_traj[0]
       
        self.fix_orn = np.array(orn)
        self.fix_orn = np.expand_dims(orn,axis=0) 


        pos = [self.obj_position[0]-0.03, self.obj_position[1]+0.2, self.obj_position[2] + 0.18]
        for i in range(30):
           self.robot.operationSpacePositionControl(pos,orn,null_pose=self.null_q,gripperPos=0)

        pos = [self.obj_position[0]-0.03, self.obj_position[1]+0.15, self.obj_position[2] + 0.075]
        for i in range(30):
           self.robot.operationSpacePositionControl(pos,orn,null_pose=self.null_q,gripperPos=0)

        pos = self.robot.getEndEffectorPos()
        pos[0] += np.random.uniform(-0.03,0.03)
        pos[1] += np.random.uniform(0,0.05)
        pos[2] += np.random.uniform(0,0.01)
        for i in range(30):
           self.robot.operationSpacePositionControl(pos,orn,null_pose=self.null_q,gripperPos=0)

        self.p.resetBasePositionAndOrientation (self.obj_id, self.obj_position, self.obj_orientation)
   
    def get_success(self,seg=None):
        pos = self.p.getBasePositionAndOrientation(self.obj_id)[0]
        if pos[1] < self.obj_position[1] - 0.1:
          self.count += 1
          r_pos = self.robot.getEndEffectorPos()
          return True
        else:
          return False
