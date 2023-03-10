#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.append('./Eval')
sys.path.append('/')

from scipy.spatial.transform import Rotation as R

from env import Engine


class Engine152(Engine):
    def __init__(self, worker_id, opti, p_id, taskId=15, maxSteps=15, n_dmps=3, cReward=True):
        super(Engine152,self).__init__(opti, wid=worker_id, p_id=p_id, maxSteps=maxSteps, taskId=taskId, n_dmps=n_dmps, cReward=cReward,robot_model=None)
        self.opti = opti
        self._wid = worker_id
        self.max_steps = maxSteps 
        self.cReward = cReward
        self.robot.gripper_max_force = 10000.0

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
        self.obj_scaling = 0.9
        self.obj_id = self.p.loadURDF(os.path.join(self.urdf_dir,"obj_libs/bottles/b1/b1.urdf"),globalScaling=self.obj_scaling)
        self.obj2_scaling = 1.0
        self.obj2_id = self.p.loadURDF(os.path.join(self.urdf_dir,"obj_libs/bottles/b2/b2.urdf"),globalScaling=self.obj2_scaling,useFixedBase=True)
        #self.p.changeVisualShape (self.obj_id, -1, rgbaColor=[0.,.8,.8,1])
        self.p.changeVisualShape (self.obj2_id, -1, rgbaColor=[1.,0.,0.,1])

    def reset_obj(self):
        self.p.resetBasePositionAndOrientation(self.obj_id,[0.3637 + 0.06, -0.05, 0.3515],[0, 0, -0.1494381, 0.9887711])

         
        self.p.resetBasePositionAndOrientation(self.obj2_id,[0.3337, 0.05, 0.3215],[0, 0, -0.1494381, 0.9887711])

        self.prevDist = 0.23

    def init_motion(self):
        self.data_q = np.load (os.path.join(self.robot_recordings_dir,"47-4/q.npy"))
        self.data_gripper = np.load (os.path.join(self.robot_recordings_dir,"47-4/gripper.npy"))
        self.robot.setJointValue(self.data_q[0],gripper=self.data_gripper[0])

    def init_grasp(self):
        qlist = np.load(os.path.join(self.robot_recordings_dir,'47-4/q.npy'))
        glist = np.load(os.path.join(self.robot_recordings_dir,'47-4/gripper.npy'))
        num_q = len(qlist[0])

        self.robot.setJointValue(qlist[0],glist[0])

        for i in range(10,len(qlist)):
            glist[i] = min(130,glist[i])
            self.robot.jointPositionControl(qlist[i],gripper=glist[i])
            #time.sleep(0.1)

        cur_joint = self.robot.get_joint_values()
        cur_pos = self.robot.getEndEffectorPos()
        cur_orn = self.robot.getEndEffectorOrn()
        pos_diff = np.random.uniform(-0.1,0.1,size=(2,))
        cur_pos[:2] = cur_pos[:2] + pos_diff
        cur_pos[2] -= -0.01
        for i in range(19):
           self.robot.operationSpacePositionControl(cur_pos,cur_orn,null_pose=cur_joint,gripperPos=130)

        self.obj2_posi = np.array(self.p.getBasePositionAndOrientation(self.obj2_id)[0])
        self.obj2_ori = self.p.getBasePositionAndOrientation(self.obj2_id)[1]
        r = R.from_quat(self.obj2_ori)
        HTrans = np.zeros((4,4))
        HTrans[:3,:3] = r.as_dcm()
        HTrans[:3,3] = self.obj2_posi

        rotation_degree = np.random.uniform(-0.5,0.5)
        addRot = R.from_rotvec(rotation_degree * np.array([0,0,1]))
        addHTrans = np.zeros((4,4))
        addHTrans[:3,:3] = addRot.as_dcm()
        NewHTrans = addHTrans.dot(HTrans)

        trans = np.random.uniform(-0.1,0.1,size=(2,))
        self.obj2_posi[:2] = self.obj2_posi[:2] + trans
        self.obj2_ori = R.from_dcm(NewHTrans[:3,:3]).as_quat()
        self.p.resetBasePositionAndOrientation(self.obj2_id,self.obj2_posi,self.obj2_ori)


        self.fix_orn = self.p.getLinkState(self.robotId, 7)[1]
        self.fix_orn = [self.fix_orn]
        self.start_pos = self.p.getLinkState (self.robotId, 7)[0]

    def get_success(self,seg=None):
        obj = self.p.get_aabb(self.obj_id)
        obj_center = [(x + y) * 0.5 for x, y in zip (obj[0], obj[1])]
        gripper_pos = self.robot.getGripperTipPos()

        dist = np.linalg.norm(np.array(obj_center) - gripper_pos)
        obj_v = self.p.getBaseVelocity(self.obj_id)[0]
        obj_v_norm = np.linalg.norm(obj_v)


        left_closet_info = self.p.getContactPoints (self.robotId, self.obj_id, self.robot.gripper_left_tip_index, -1)
        right_closet_info = self.p.getContactPoints (self.robotId, self.obj_id, self.robot.gripper_right_tip_index, -1)
        cc = self.p.getContactPoints(self.obj_id,self.obj2_id)
        if len (left_closet_info)==0 and len (right_closet_info)==0 and len(cc) > 0 and obj_v_norm > 1.0 and dist > 0.05:
           return True
        else:
           return False 
