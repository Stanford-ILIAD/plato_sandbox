#!/usr/bin/env python3

import math
import os
import sys

import numpy as np

sys.path.append('./Eval')
sys.path.append('/')

from env import Engine
from sbrl.envs.bullet_envs.utils_env import safe_path


class Engine121(Engine):
    def __init__(self, worker_id, opti, p_id, taskId=5, maxSteps=15, n_dmps=3, cReward=True):
        super(Engine121,self).__init__(opti, wid=worker_id, p_id=p_id, maxSteps=maxSteps, taskId=taskId, n_dmps=n_dmps, cReward=cReward,robot_model=None)
        self.opti = opti
        self._wid = worker_id
        self.robot.gripper_max_force = 200.0
        self.robot.arm_max_force = 200.0
        self.robot.jd = [0.01] * 14

        self.p.setPhysicsEngineParameter(enableConeFriction=1)
        self.p.setPhysicsEngineParameter(contactBreakingThreshold=0.001)
        self.p.setPhysicsEngineParameter(allowedCcdPenetration=0.0)

        self.p.setPhysicsEngineParameter(numSolverIterations=20)
        self.p.setPhysicsEngineParameter(numSubSteps=10)

        self.p.setPhysicsEngineParameter(constraintSolverType=self.p.CONSTRAINT_SOLVER_LCP_DANTZIG,globalCFM=0.000001)
        self.p.setPhysicsEngineParameter(enableFileCaching=0)

        self.p.setGravity(0,0,-9.81)


    def reset_new(self):
        self.log_path = safe_path(os.path.join(self.log_root,'epoch-{}'.format(self.epoch_num)))
        self.log_info = open(os.path.join(self.log_root,'epoch-{}.txt'.format(self.epoch_num)),'w')
        self.seq_num = 0
        self.init_dmp()
        self.init_motion ()
        self.init_rl ()
        self.reset_obj ()
        self.init_grasp ()
      
        return self.get_observation()

    def init_obj(self):
        self.obj1_scaling = 1.2
        self.obj_id = self.p.loadURDF( os.path.join(self.resources_dir, "urdf/obj_libs/bottles/b1/b1.urdf"),globalScaling=self.obj1_scaling)
        self.obj2_scaling = 2.0
        self.obj2_id = self.p.loadURDF( os.path.join(self.resources_dir, "urdf/obj_libs/bottles/b6/b6.urdf"),globalScaling=self.obj2_scaling)
        self.p.changeVisualShape (self.obj_id, -1, rgbaColor=[1.,0.,0.,1])
        self.p.changeVisualShape (self.obj2_id, -1, rgbaColor=[0.3,0.3,0.9,1])
   
    def reset_obj(self):
        self.obj1_orn = self.p.getQuaternionFromEuler([math.pi/2.0,0,0])

        obj_x = 0.31
        obj_y = 0.03
        transl = np.random.uniform(-0.1,0.1,size=(2,))
        self.obj_position = np.array([obj_x+transl[0],obj_y+transl[1],0.3])
        self.p.resetBasePositionAndOrientation(self.obj_id,self.obj_position,self.obj1_orn)
        self.p.resetBasePositionAndOrientation(self.obj2_id,[0.3796745705777075, -0.16140520662377353 + 0.05, 0.40416254394211476-0.03],[0, 0, -0.1494381, 0.9887711])
  
        obj_friction_ceof = 10.0
        self.p.changeDynamics(self.obj_id, -1, mass=1.9)
        self.p.changeDynamics(self.obj_id, -1, lateralFriction=obj_friction_ceof)
        self.p.changeDynamics(self.obj_id, -1, rollingFriction=obj_friction_ceof)
        self.p.changeDynamics(self.obj_id, -1, spinningFriction=obj_friction_ceof)
        self.p.changeDynamics(self.obj_id, -1, linearDamping=40.0)
        self.p.changeDynamics(self.obj_id, -1, angularDamping=1.0)
        self.p.changeDynamics(self.obj_id, -1, contactStiffness=1.0, contactDamping=0.9)


        obj2_friction_ceof = 10.0
        self.p.changeDynamics(self.obj2_id, -1, mass=1.9)
        self.p.changeDynamics(self.obj2_id, -1, lateralFriction=obj2_friction_ceof)
        self.p.changeDynamics(self.obj2_id, -1, rollingFriction=obj2_friction_ceof)
        self.p.changeDynamics(self.obj2_id, -1, spinningFriction=obj2_friction_ceof)
        self.p.changeDynamics(self.obj2_id, -1, linearDamping=20.0)
        self.p.changeDynamics(self.obj2_id, -1, angularDamping=1.0)
        self.p.changeDynamics(self.obj2_id, -1, contactStiffness=1.0, contactDamping=0.9)

        table_friction_ceof = 0.4
        self.p.changeDynamics(self.table_id, -1, lateralFriction=table_friction_ceof)
        self.p.changeDynamics(self.table_id, -1, rollingFriction=table_friction_ceof)
        self.p.changeDynamics(self.table_id, -1, spinningFriction=table_friction_ceof)
        self.p.changeDynamics(self.table_id, -1, contactStiffness=1.0, contactDamping=0.9)

    def init_motion(self):
        self.data_q = np.load (os.path.join(self.robot_recordings_dir,"29-0/q.npy"))
        self.data_gripper = np.load(os.path.join(self.robot_recordings_dir,"29-0/gripper.npy"))#np.load (self.env_root + '/init/gripper.npy')
        self.robot.setJointValue(self.data_q[0],gripper=self.data_gripper[0])

    def init_grasp(self):
        self.robot.gripper_control(0)

        qlist = np.load( os.path.join(self.robot_recordings_dir, "29-1/q.npy"))
        glist = np.load( os.path.join(self.robot_recordings_dir, "29-1/gripper.npy"))
        num_q = len(qlist[0])

        self.robot.setJointValue(qlist[0],glist[0])
        for i in range(0,5,1):
            glist[i] = max(200,glist[i])
            self.robot.jointPositionControl(qlist[i],gripper=glist[i])

        self.null_q = self.robot.get_joint_values()
        pos = self.robot.getEndEffectorPos()
        pos[0] += -.13
        pos[1] += 0.18
        pos[2] += -.01
        orn = self.robot.getEndEffectorOrn()
        for i in range(109):
           self.robot.operationSpacePositionControl(pos,orn,null_pose=self.null_q,gripperPos=200)
 
        self.fix_orn = self.p.getLinkState(self.robotId, 7)[1]
        self.fix_orn = [self.fix_orn]
        self.start_pos = self.p.getLinkState (self.robotId, 7)[0]


    def get_success(self,seg=None):
        objid_pixel = np.sum(seg == self.obj_id)
        if objid_pixel > 200:
          return True
        else:
          return False

