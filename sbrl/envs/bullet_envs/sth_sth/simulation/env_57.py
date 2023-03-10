#!/usr/bin/env python3
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.append('./Eval')
sys.path.append('/')

from env import Engine


class Engine57(Engine):
    def __init__(self, worker_id, opti, p_id, taskId=5, maxSteps=15, n_dmps=3, cReward=True):
        super(Engine57,self).__init__(opti, wid=worker_id, p_id=p_id, maxSteps=maxSteps, taskId=taskId, n_dmps=n_dmps, cReward=cReward,robot_model=None)
        self.opti = opti
        self._wid = worker_id
        self.robot.gripper_max_force = 200.0
        self.robot.arm_max_force = 200.0
        self.robot.jd = [0.01] * 14

        self.p = p_id
        self.p.setPhysicsEngineParameter(enableConeFriction=1)
        self.p.setPhysicsEngineParameter(contactBreakingThreshold=0.001)
        self.p.setPhysicsEngineParameter(allowedCcdPenetration=0.0)

        self.p.setPhysicsEngineParameter(numSolverIterations=20)
        self.p.setPhysicsEngineParameter(numSubSteps=10)

        self.p.setPhysicsEngineParameter(constraintSolverType=self.p.CONSTRAINT_SOLVER_LCP_DANTZIG,globalCFM=0.000001)
        self.p.setPhysicsEngineParameter(enableFileCaching=0)

        self.p.setTimeStep(1 / 30.0)
        self.p.setGravity(0,0,-9.81)


    def init_obj(self):
        obj_x = 0.42
        obj_y = -0.01
        obj_z = 0.39
        self.obj_position = [0.35, -0.02, obj_z]
        self.obj_scaling = 1.0
        self.obj_orientation = self.p.getQuaternionFromEuler([0, 0, 0])
        self.obj_id = self.p.loadURDF(os.path.join(self.urdf_dir,"obj_libs/bottles/b7/b7.urdf"),basePosition=self.obj_position,baseOrientation=self.obj_orientation,globalScaling=self.obj_scaling)

        #self.p.changeVisualShape (self.obj_id, -1, rgbaColor=[1.,0.,0.,1])
        #self.p.addUserDebugLine([0,0,0],[0.1,0,0],[1,0,0],parentObjectUniqueId=self.obj_id, parentLinkIndex=-1)
        #self.p.addUserDebugLine([0,0,0],[0,0.1,0],[0,1,0],parentObjectUniqueId=self.obj_id, parentLinkIndex=-1)
        #self.p.addUserDebugLine([0,0,0],[0,0,0.1],[0,0,1],parentObjectUniqueId=self.obj_id, parentLinkIndex=-1)

    def reset_obj(self):
        obj_x = 0.42
        obj_y = -0.01
        transl = np.random.uniform(-0.1,0.1,size=(2,))
        self.obj_x = obj_x + transl[0]
        self.obj_y = obj_y + transl[1]
        self.obj_position = np.array([self.obj_x,self.obj_y,0.39])

        self.p.resetBasePositionAndOrientation(self.obj_id,self.obj_position,self.obj_orientation)
        
        obj_friction_ceof = 1.0
        self.p.changeDynamics(self.obj_id, -1, mass=20.0)
        self.p.changeDynamics(self.obj_id, -1, lateralFriction=obj_friction_ceof)
        self.p.changeDynamics(self.obj_id, -1, rollingFriction=obj_friction_ceof)
        self.p.changeDynamics(self.obj_id, -1, spinningFriction=obj_friction_ceof)
        #self.p.changeDynamics(self.obj_id, -1, linearDamping=40.0)
        #self.p.changeDynamics(self.obj_id, -1, angularDamping=100.0)
        #self.p.changeDynamics(self.obj_id, -1, contactStiffness=1000.0, contactDamping=0.9)

        #table_friction_ceof = 1.0
        #self.p.changeDynamics(self.table_id, -1, lateralFriction=table_friction_ceof)
        #self.p.changeDynamics(self.table_id, -1, rollingFriction=table_friction_ceof)
        #self.p.changeDynamics(self.table_id, -1, spinningFriction=table_friction_ceof)
        #self.p.changeDynamics(self.table_id, -1, contactStiffness=1000.0, contactDamping=10.0)

        for i in range(100):
          self.p.stepSimulation()

    def init_grasp(self):
        self.robot.gripper_control(255)
  
        pos_traj = np.load (os.path.join (self.configs_dir, 'init', 'pos.npy'))
        orn_traj = np.load (os.path.join (self.configs_dir, 'init', 'orn.npy'))
        self.fix_orn = np.load (os.path.join (self.configs_dir, 'init', 'orn.npy'))

        for j in range (7):
            self.p.resetJointState(self.robotId, j, self.data_q[0][j], self.data_dq[0][j])

        for init_t in range(100):
            box = self.p.get_aabb(self.obj_id)
            center = [(x+y)*0.5 for x,y in zip(box[0],box[1])]
            center[0] -= 0.05
            center[1] -= 0.05
            center[2] += 0.03
            # center = (box[0]+box[1])*0.5
        points = np.array ([pos_traj[0], center])

        self.null_q = self.robot.get_joint_values()#self.initial_pos
    
        self.obj_x, self.obj_y, self.obj_z = self.obj_position
        pos = [self.obj_x-0.03,self.obj_y-0.2,self.obj_z+0.1]
        orn = self.robot.getEndEffectorOrn()
        for i in range(19):
           self.robot.operationSpacePositionControl(pos,orn,null_pose=self.null_q,gripperPos=220)

        start_id = 0
        self.p.resetBasePositionAndOrientation(self.obj_id,self.obj_position,self.obj_orientation)
 
        self.start_pos = self.p.getLinkState (self.robotId, 7)[0]

    def taskColliDet(self):
        colli = False
        for y in [0,1,2,3,4,5,6]:
          c = self.p.getContactPoints(bodyA=self.obj_id,bodyB=self.robotId,linkIndexB=y)
          if len(c) > 0:
            colli = True
            return True
        return False

    def get_success (self,seg=None):
        orn1 = self.p.getBasePositionAndOrientation(self.obj_id)[1]
        r = R.from_quat(orn1)
        vec = r.as_rotvec()
        rotnorm = np.linalg.norm(vec)
        vec = vec / rotnorm
        print("vec",vec,"angle",np.arccos(np.abs(vec[2]))/np.pi * 180.0,"rotnorm",rotnorm/np.pi * 180)
        if np.arccos(np.abs(vec[2]))/np.pi * 180.0 > 70.0 and rotnorm /np.pi * 180 > 30.0:
          return True
        else:
          return False
