import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from sbrl.utils.transform_utils import euler2quat_ext


def draw_point_gui(position, color, client: int, radius=-0.005):
    sphere_vis = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=radius, rgbaColor=color,
                                     physicsClientId=client)
    sphere_body = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1,
                                    baseVisualShapeIndex=sphere_vis,
                                    basePosition=position,
                                    useMaximalCoordinates=False, physicsClientId=client)
    return sphere_vis, sphere_body


def draw_box_gui(position, orientation, size, color, client: int):
    box_vis = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=np.asarray(size) / 2, rgbaColor=color,
                                  physicsClientId=client)
    box_body = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1,
                                 baseVisualShapeIndex=box_vis,
                                 basePosition=position,
                                 baseOrientation=orientation,
                                 useMaximalCoordinates=False, physicsClientId=client)
    return box_vis, box_body


def draw_mug_gui(position, orientation, size, color, client: int):
    cyl_vis = p.createVisualShape(shapeType=p.GEOM_CYLINDER, radius=size[2] * 0.75 / 2, length=size[2], rgbaColor=color,
                                  physicsClientId=client)
    box_extents = np.array([size[2] * 0.1, size[2] * 0.5, size[2]])
    base_pos = position + Rotation.from_quat(orientation).apply(np.array([0., 0., size[2] / 2]))
    box_vis = p.createVisualShape(shapeType=p.GEOM_BOX,
                                  halfExtents=box_extents / 2, rgbaColor=color,
                                  physicsClientId=client)
    mug_body = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1,
                                 baseVisualShapeIndex=cyl_vis,
                                 basePosition=base_pos,
                                 baseOrientation=orientation,
                                 linkMasses=[0],
                                 linkCollisionShapeIndices=[-1],
                                 linkVisualShapeIndices=[box_vis],
                                 linkPositions=[[0., size[1] / 2 - box_extents[1] / 4, 0.]],
                                 linkOrientations=[[0, 0, 0, 1.]],
                                 linkInertialFramePositions=[[0, 0, 0]],
                                 linkInertialFrameOrientations=[[0, 0, 0, 1]],
                                 linkParentIndices=[0],
                                 linkJointTypes=[p.JOINT_FIXED],
                                 linkJointAxis=[[0, 0, 0]],
                                 physicsClientId=client)
    return [cyl_vis, box_vis], mug_body


def draw_cyl_gui(position, orientation, radius, length, color, client: int):
    cyl_vis = p.createVisualShape(shapeType=p.GEOM_CYLINDER, radius=radius, length=length, rgbaColor=color,
                                  physicsClientId=client)
    cyl_body = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1,
                                 baseVisualShapeIndex=cyl_vis,
                                 basePosition=position,
                                 baseOrientation=orientation,
                                 useMaximalCoordinates=False, physicsClientId=client)
    return cyl_vis, cyl_body


""" stuff from https://github.com/danfeiX/deftenv """
NULL_ID = -1


def unit_point():
    return (0, 0, 0)


def unit_quat():
    return (0, 0, 0, 1)


def unit_pose():
    return (unit_point(), unit_quat())


def create_visual_shape(geometry, pose=unit_pose(), color=(1, 0, 0, 1), specular=None, client=0):
    if (color is None):  # or not has_gui():
        return NULL_ID
    point, quat = pose
    visual_args = {
        'rgbaColor': color,
        'visualFramePosition': point,
        'visualFrameOrientation': quat,
        'physicsClientId': client,
    }
    visual_args.update(geometry)
    if specular is not None:
        visual_args['specularColor'] = specular
    return p.createVisualShape(**visual_args)


def create_collision_shape(geometry, pose=unit_pose(), client=0):
    point, quat = pose
    collision_args = {
        'collisionFramePosition': point,
        'collisionFrameOrientation': quat,
        'physicsClientId': client,
    }
    collision_args.update(geometry)
    if 'length' in collision_args:
        # TODO: pybullet bug visual => length, collision => height
        collision_args['height'] = collision_args['length']
        del collision_args['length']
    return p.createCollisionShape(**collision_args)


def create_shape(geometry, pose=unit_pose(), collision=True, client=0, **kwargs):
    collision_id = create_collision_shape(geometry, pose=pose, client=client) if collision else NULL_ID
    visual_id = create_visual_shape(geometry, pose=pose, client=client, **kwargs)
    return collision_id, visual_id


def get_box_geometry(width, length, height):
    return {
        'shapeType': p.GEOM_BOX,
        'halfExtents': [width / 2., length / 2., height / 2.]
    }


class Object(object):
    """Borrowed from iGibson (https://github.com/StanfordVL/iGibson)"""

    def __init__(self):
        self.body_id = None
        self.loaded = False

    def load(self):
        if self.loaded:
            return self.body_id
        self.body_id = self._load()
        self.loaded = True
        return self.body_id

    def get_position_orientation(self):
        return p.getBasePositionAndOrientation(self.body_id)

    def get_position(self):
        pos, _ = p.getBasePositionAndOrientation(self.body_id)
        return pos

    def get_orientation(self):
        """Return object orientation
        :return: quaternion in xyzw
        """
        _, orn = p.getBasePositionAndOrientation(self.body_id)
        return orn

    def set_position(self, pos):
        _, old_orn = p.getBasePositionAndOrientation(self.body_id)
        p.resetBasePositionAndOrientation(self.body_id, pos, old_orn)

    def set_orientation(self, orn):
        old_pos, _ = p.getBasePositionAndOrientation(self.body_id)
        p.resetBasePositionAndOrientation(self.body_id, old_pos, orn)

    def set_position_orientation(self, pos, orn):
        p.resetBasePositionAndOrientation(self.body_id, pos, orn)

    def clean(self, clientId=0):
        if self.loaded:
            p.removeBody(self.body_id, physicsClientId=clientId)


class Hook(Object):
    def __init__(self, width, length1, length2, color=(0, 1, 0, 1)):
        super(Hook, self).__init__()
        self._width = width
        self._length1 = length1
        self._length2 = length2
        self._color = color

    def load(self, clientId=0):
        self.loaded = True

        collision_id1, visual_id1 = create_shape(
            get_box_geometry(self._length1, self._width, self._width), color=self._color, client=clientId)
        collision_id2, visual_id2 = create_shape(
            get_box_geometry(self._length2, self._width, self._width), color=self._color, client=clientId)
        self.body_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=collision_id1,
            baseVisualShapeIndex=visual_id1,
            basePosition=unit_point(),
            baseOrientation=unit_quat(),
            baseInertialFramePosition=unit_point(),
            baseInertialFrameOrientation=unit_quat(),
            linkMasses=(0.5,),
            linkCollisionShapeIndices=[collision_id2],
            linkVisualShapeIndices=[visual_id2],
            linkPositions=[(-self._length1 / 2 + self._width / 2, -self._length2 / 2 + self._width / 2, 0)],
            linkOrientations=[euler2quat_ext(0, 0, np.pi / 2)],
            linkInertialFramePositions=[(0, 0, 0)],
            linkInertialFrameOrientations=[unit_quat()],
            linkParentIndices=[0],
            linkJointTypes=[p.JOINT_FIXED],
            linkJointAxis=[[0, 0, 0]],
            physicsClientId=clientId,
        )
