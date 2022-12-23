import numpy as np

from sbrl.envs.spec import Spec
from sbrl.models.rrt.utilities.geometry import es_points_along_line
from sbrl.utils.python_utils import AttrDict
from sbrl.utils.torch_utils import concatenate


class NormalizedCollisionSearchSpace:
    def __init__(self, collision_fn, search_spec: Spec):
        """
        Initialize Search Space, presents -1 to 1 to algorithm
        """

        self.get_collision_fn = collision_fn
        self.search_spec = search_spec

        self.dimensions = len(self.sample())
        lower, upper = self.search_spec.limits(search_spec.all_names)
        self.lower = np.concatenate([l.flatten() for l in lower.tolist()])  # flat
        self.upper = np.concatenate([u.flatten() for u in upper.tolist()])  # flat

        self.dimension_lengths = np.stack([self.lower, self.upper], axis=-1)  # create new axis


    def normalize(self, d_un: AttrDict):
        return self.search_spec.scale_to_unit_box(d_un, d_un.leaf_keys())

    def unnormalize(self, d_n: AttrDict):
        return self.search_spec.scale_from_unit_box(d_n, d_n.leaf_keys())

    def parse_x_to_spec(self, x):
        return self.search_spec.parse_from_concatenated_flat(np.asarray(x), self.search_spec.all_names)

    def parse_spec_to_x(self, d: AttrDict):
        return concatenate(d.leaf_apply(lambda arr: arr.flatten()), self.search_spec.all_names)

    def obstacle_free(self, x):
        """
        Check if a location resides inside of an obstacle
        :param x: location to check
        :return: True if not inside an obstacle, False otherwise
        """
        return self.get_collision_fn(self.parse_x_to_spec(x))

    def sample_free(self):
        """
        Sample a location within X_free
        :return: random location within X_free
        """
        while True:  # sample until not inside of an obstacle
            d = self.sample_spec()
            if self.get_collision_fn(d):
                return self.parse_spec_to_x(d)

    def collision_free(self, start, end, r):
        """
        Check if a line segment intersects an obstacle
        :param start: starting point of line
        :param end: ending point of line
        :param r: resolution of points to sample along edge when checking for collisions
        :return: True if line segment does not intersect an obstacle, False otherwise
        """
        points = es_points_along_line(start, end, r)
        coll_free = all(map(self.obstacle_free, points))
        return coll_free

    def sample_spec(self):
        uniform_dict = self.search_spec.get_uniform(self.search_spec.all_names, 1).leaf_apply(lambda arr: arr[0])
        dc = self.normalize(uniform_dict)
        return dc

    def sample(self):
        """
        Return a random location within X
        :return: random location within X (not necessarily X_free)
        """
        dc = self.sample_spec()
        return self.parse_spec_to_x(dc)
