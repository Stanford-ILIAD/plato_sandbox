import random

import numpy as np
import pymesh
import torch
from scipy.spatial.transform import Rotation as R

import sbrl.utils.transform_utils as T
from sbrl.utils.control_utils import batch_orientation_error
from sbrl.utils.np_utils import clip_norm
from sbrl.utils.torch_utils import to_numpy, to_torch


class SmoothNoise:
    def __init__(self, dims, trajectory_length=100, gen_period=5, pref_dir=None, alpha=0.5, decay_steps=5,
                 generator=None, max_norm=np.inf):
        """
        Generates smooth noise along a trajectory that terminates at zero.

        Parameters
        ----------
        dims: how many dimensions to generate noise for
        trajectory_length: how many time steps to generate
        gen_period: period between noise generator calls.
        pref_dir: |v| = dims, the direction to move TODO
        alpha: the smoothing coefficient for the added noise.
        decay_steps: when to start decaying.
        generator: what generates the noise
        max_norm: how big can the norm get

        """
        self._dims = dims
        self._trajectory_length = trajectory_length
        self._gen_period = gen_period

        self._pref_dir = pref_dir
        self._alpha = alpha
        self._decay_steps = decay_steps

        self._last_noise = np.zeros(self._dims)
        self._last_noise_target = np.zeros(self._dims)

        self._generator = generator
        if generator is None:
            self._generator = lambda: np.random.rand(dims)

        self._max_norm = max_norm

    def __call__(self, idx, decay=None):
        steps_left = max(self._trajectory_length - idx, 1)
        if decay:
            self._last_noise *= decay
            return self._last_noise
        elif steps_left <= self._decay_steps:
            # decay to zero without adding more noise, linearly
            return self._last_noise * (steps_left - 1) / self._decay_steps
        else:
            if idx % self._gen_period == 0:
                self._last_noise_target += self._generator()
                self._last_noise_target = clip_norm(self._last_noise_target, self._max_norm)

            self._last_noise = self._alpha * self._last_noise_target + (1 - self._alpha) * self._last_noise
            return self._last_noise.copy()


# Create a maze using the depth-first algorithm described at
# https://scipython.com/blog/making-a-maze/
# Christian Hill, April 2017.
class Cell:
    """A cell in the maze.

    A maze "Cell" is a point in the grid which may be surrounded by walls to
    the north, east, south or west.

    """

    # A wall separates a pair of cells in the N-S or W-E directions.
    wall_pairs = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}

    def __init__(self, x, y):
        """Initialize the cell at (x,y). At first it is surrounded by walls."""

        self.x, self.y = x, y
        self.walls = {'N': True, 'S': True, 'E': True, 'W': True}

    def has_all_walls(self):
        """Does this cell still have all its walls?"""

        return all(self.walls.values())

    def knock_down_wall(self, other, wall):
        """Knock down the wall between cells self and other."""

        self.walls[wall] = False
        other.walls[Cell.wall_pairs[wall]] = False

    def cell_to_np(self):
        return self.walls["N"] * 8 + self.walls["W"] * 4 + self.walls["S"] * 2 + self.walls["E"] * 1

    def cell_from_np(self, cell_np):
        assert cell_np <= 15, "Invalid cell size"
        self.walls["N"] = bool(cell_np // 8)
        self.walls["W"] = bool((cell_np % 8) // 4)
        self.walls["S"] = bool((cell_np % 4) // 2)
        self.walls["E"] = bool((cell_np % 2))

    def __eq__(self, other):
        assert self.cell_to_np() == other.cell_to_np()


class Maze:
    """A Maze, represented as a grid of cells."""

    def __init__(self, nx, ny, ix=0, iy=0, ):
        """Initialize the maze grid.
        The maze consists of nx x ny cells and will be constructed starting
        at the cell indexed at (ix, iy).

        """

        self.nx, self.ny = nx, ny
        self.ix, self.iy = ix, iy
        self.maze_map = [[Cell(x, y) for y in range(ny)] for x in range(nx)]

    def cell_at(self, x, y):
        """Return the Cell object at (x,y)."""

        return self.maze_map[x][y]

    def get_contiguous_y_walls(self, column):
        """
        columns go from 0 ... ny

        returns a wall set list of tuples(x1, x2) of inclusive cell wall lengths
        """
        cell_y = min(column, self.ny - 1)
        wall_side = 'E' if column == self.ny else 'W'

        all_walls = []
        curr_wall = []
        for x in range(self.nx):
            if self.cell_at(x, cell_y).walls[wall_side]:
                curr_wall.append(x)
            else:
                if len(curr_wall) > 0:
                    all_walls.append((curr_wall[0], curr_wall[-1]))
                curr_wall.clear()

        if len(curr_wall) > 0:
            all_walls.append((curr_wall[0], curr_wall[-1]))
        return all_walls

    def get_contiguous_x_walls(self, row):
        """
        columns go from 0 ... ny

        returns a wall set list of tuples(x1, x2) of inclusive cell wall lengths
        """
        row_x = min(row, self.nx - 1)
        wall_side = 'S' if row == self.nx else 'N'

        all_walls = []
        curr_wall = []
        for y in range(self.ny):
            if self.cell_at(row_x, y).walls[wall_side]:
                curr_wall.append(y)
            else:
                if len(curr_wall) > 0:
                    all_walls.append((curr_wall[0], curr_wall[-1]))
                curr_wall.clear()

        if len(curr_wall) > 0:
            all_walls.append((curr_wall[0], curr_wall[-1]))
        return all_walls

    def __str__(self):
        """Return a (crude) string representation of the maze."""

        maze_rows = ['-' * self.nx * 2]
        for y in range(self.ny):
            maze_row = ['|']
            for x in range(self.nx):
                if self.maze_map[x][y].walls['E']:
                    maze_row.append(' |')
                else:
                    maze_row.append('  ')
            maze_rows.append(''.join(maze_row))
            maze_row = ['|']
            for x in range(self.nx):
                if self.maze_map[x][y].walls['S']:
                    maze_row.append('-+')
                else:
                    maze_row.append(' +')
            maze_rows.append(''.join(maze_row))
        return '\n'.join(maze_rows)

    def write_svg(self, filename):
        """Write an SVG image of the maze to filename."""

        aspect_ratio = self.nx / self.ny
        # Pad the maze all around by this amount.
        padding = 10
        # Height and width of the maze image (excluding padding), in pixels
        height = 500
        width = int(height * aspect_ratio)
        # Scaling factors mapping maze coordinates to image coordinates
        scy, scx = height / self.ny, width / self.nx

        def write_wall(ww_f, ww_x1, ww_y1, ww_x2, ww_y2):
            """Write a single wall to the SVG image file handle f."""

            print('<line x1="{}" y1="{}" x2="{}" y2="{}"/>'
                  .format(ww_x1, ww_y1, ww_x2, ww_y2), file=ww_f)

        # Write the SVG image file for maze
        with open(filename, 'w') as f:
            # SVG preamble and styles.
            print('<?xml version="1.0" encoding="utf-8"?>', file=f)
            print('<svg xmlns="http://www.w3.org/2000/svg"', file=f)
            print('    xmlns:xlink="http://www.w3.org/1999/xlink"', file=f)
            print('    width="{:d}" height="{:d}" viewBox="{} {} {} {}">'
                  .format(width + 2 * padding, height + 2 * padding,
                          -padding, -padding, width + 2 * padding, height + 2 * padding),
                  file=f)
            print('<defs>\n<style type="text/css"><![CDATA[', file=f)
            print('line {', file=f)
            print('    stroke: #000000;\n    stroke-linecap: square;', file=f)
            print('    stroke-width: 5;\n}', file=f)
            print(']]></style>\n</defs>', file=f)
            # Draw the "South" and "East" walls of each cell, if present (these
            # are the "North" and "West" walls of a neighbouring cell in
            # general, of course).
            for x in range(self.nx):
                for y in range(self.ny):
                    if self.cell_at(x, y).walls['S']:
                        x1, y1, x2, y2 = x * scx, (y + 1) * scy, (x + 1) * scx, (y + 1) * scy
                        write_wall(f, x1, y1, x2, y2)
                    if self.cell_at(x, y).walls['E']:
                        x1, y1, x2, y2 = (x + 1) * scx, y * scy, (x + 1) * scx, (y + 1) * scy
                        write_wall(f, x1, y1, x2, y2)
            # Draw the North and West maze border, which won't have been drawn
            # by the procedure above.
            print('<line x1="0" y1="0" x2="{}" y2="0"/>'.format(width), file=f)
            print('<line x1="0" y1="0" x2="0" y2="{}"/>'.format(height), file=f)
            print('</svg>', file=f)

    def find_valid_neighbours(self, cell):
        """Return a list of unvisited neighbours to cell."""

        delta = [('W', (-1, 0)),
                 ('E', (1, 0)),
                 ('S', (0, 1)),
                 ('N', (0, -1))]
        neighbours = []
        for direction, (dx, dy) in delta:
            x2, y2 = cell.x + dx, cell.y + dy
            if (0 <= x2 < self.nx) and (0 <= y2 < self.ny):
                neighbour = self.cell_at(x2, y2)
                if neighbour.has_all_walls():
                    neighbours.append((direction, neighbour))
        return neighbours

    def find_neighbours(self, cell):
        """Return a list of unvisited neighbours to cell."""

        delta = [('W', (-1, 0)),
                 ('E', (1, 0)),
                 ('S', (0, 1)),
                 ('N', (0, -1))]
        neighbours = []
        for direction, (dx, dy) in delta:
            x2, y2 = cell.x + dx, cell.y + dy
            if (0 <= x2 < self.nx) and (0 <= y2 < self.ny):
                neighbour = self.cell_at(x2, y2)
                neighbours.append((direction, neighbour))
        return neighbours

    def make_maze(self, open=False):
        # Total number of cells.
        n = self.nx * self.ny
        cell_stack = []
        current_cell = self.cell_at(self.ix, self.iy)
        # Total number of visited cells during maze construction.
        nv = 1

        while nv < n:
            neighbours = self.find_valid_neighbours(current_cell)

            if not neighbours:
                # We've reached a dead end: backtrack.
                current_cell = cell_stack.pop()
                continue

            # Choose a random neighbouring cell and move to it.
            direction, next_cell = random.choice(neighbours)
            current_cell.knock_down_wall(next_cell, direction)
            cell_stack.append(current_cell)
            current_cell = next_cell
            nv += 1

        if open:
            raise NotImplementedError
            # # make sure no maze is closed from neighbors
            # for x in range(self.nx):
            #     for y in range(self.ny):
            #         cell = self.cell_at(x, y)
            #         check = ['N', 'W', 'S', 'E']
            #         if x == 0:
            #             check.remove('W')
            #         if x == self.nx - 1:
            #             check.remove('E')
            #         if y == 0:
            #             check.remove('N')
            #         if y == self.ny - 1:
            #             check.remove('S')
            #         # print(x, y, check, cell.walls)
            #         if all(cell.walls[c] for c in check):
            #             direction, next_cell = random.choice(self.find_neighbours(cell))
            #             cell.knock_down_wall(next_cell, direction)


    def to_numpy(self):
        maze_np = np.zeros((self.nx, self.ny), dtype=np.uint8)
        for x in range(self.nx):
            for y in range(self.ny):
                maze_np[x, y] = self.cell_at(x, y).cell_to_np()
        return maze_np

    def from_numpy(self, maze_np):
        for x in range(self.nx):
            for y in range(self.ny):
                self.cell_at(x, y).cell_from_np(maze_np[x, y])

        return maze_np

    def __eq__(self, other):
        return (self.to_numpy() == other.to_numpy()).all()


class CoordinateFrame:
    def __init__(self, parent, parent_to_child_rot=None, child_origin=None):
        assert isinstance(parent, CoordinateFrame) or parent is None, parent
        self.parent = parent
        # precomputed at beginning
        if parent is not None:
            assert isinstance(parent_to_child_rot, R)
            assert len(parent_to_child_rot.as_quat().shape) == 1, "no frames with batch rotations!"
            assert isinstance(child_origin, np.ndarray)
            assert len(child_origin.shape) == 1, "no frames with batch positions!"
            # parent to child will always map the child coordinate frame to the parent coordinate frame,
            #   but maps points in the parent frame to points in the child frame
            self.c2p_R = parent_to_child_rot.inv()  # AKA basis for child frame in parent frame
            self.c_origin_in_p = child_origin.copy()
            self.p2g_R, self.p_origin_in_g = self.parent.get_transform_to_global()
            self.c_origin_in_g = self.p2g_R.apply(self.c_origin_in_p) + self.p_origin_in_g
            self.c2g_R = self.p2g_R * self.c2p_R
        else:
            self.c_origin_in_g = np.zeros(3)
            self.c2g_R = R.from_matrix(np.eye(3))

    def get_parent(self):
        return self.parent

    def get_basis_vectors_of_frame(self, frame):
        f2s, f_in_s = CoordinateFrame.transform_from_a_to_b(frame, self)
        f2s_mat = f2s.as_matrix()

        # columns of frame -> self are the basis vectors of frame in self (x,y,z)
        columns = np.hsplit(f2s_mat, 3)
        return tuple([c.flatten() for c in columns])

    # global -> child
    def get_transform_to_global(self):
        return self.c2g_R, self.c_origin_in_g

    def transform_to_parent(self, point_in_c):
        assert self.parent is not None
        return self.c2p_R.apply(point_in_c) + self.c_origin_in_p

    def inv(self):
        return CoordinateFrame(world_frame_3D, self.c2g_R, - self.c2g_R.apply(self.c_origin_in_g, inverse=True))

    @staticmethod
    def transform_from_a_to_b(frameA, frameB):
        assert isinstance(frameA, CoordinateFrame) and isinstance(frameB, CoordinateFrame)

        a2g, a_origin_in_g = frameA.get_transform_to_global()
        b2g, b_origin_in_g = frameB.get_transform_to_global()

        # a2b = b2g.inv() * a2g
        # # delta vector is rotated from g to a frame
        # a_origin_in_b = b2g.apply(a_origin_in_g - b_origin_in_g, inverse=True)
        return CoordinateFrame.relative_frame_a_to_b(a2g, a_origin_in_g, b2g, b_origin_in_g)

    @staticmethod
    def point_from_a_to_b(pointA, frameA, frameB):

        a2g, a_origin_in_g = frameA.get_transform_to_global()
        b2g, b_origin_in_g = frameB.get_transform_to_global()

        return b2g.apply(a2g.apply(pointA) + a_origin_in_g - b_origin_in_g, inverse=True)

    @staticmethod
    def rotate_from_a_to_b(vecA, frameA, frameB):

        a2g, a_origin_in_g = frameA.get_transform_to_global()
        b2g, b_origin_in_g = frameB.get_transform_to_global()

        return b2g.apply(a2g.apply(vecA), inverse=True)

    def view_from_frame(self, frame):
        # we need s2f = w2f * s2w, and s_in_f
        return world_frame_3D.apply_a_to_b(frame, self)

    def apply_a_to_b(self, frameA, frameB):
        # self transform by frameA -> frameB, returns a new coordinate frame
        a2b, a_origin_in_b = CoordinateFrame.transform_from_a_to_b(frameA, frameB)

        # coordinate frame relative to self
        return CoordinateFrame(self, a2b, a2b.apply(-a_origin_in_b, inverse=True))

    def apply_a_to_b_local(self, frameA, frameB):
        # self transform by LOCAL frameA -> frameB, returns a new coordinate frame
        a2b, a_origin_in_b = self.transform_from_a_to_b_local(frameA, frameB)

        # coordinate frame relative to self
        return CoordinateFrame(self, a2b, a2b.apply(-a_origin_in_b, inverse=True))

    def transform_from_a_to_b_local(self, frameA, frameB):
        # self transform by frameA -> frameB, from the perspective of the local frame(self)
        # ignores the effect of whatever relative rotation between frameA and self is.
        # e.g. computing rotation matrix of an end effector in global frame's view, to get global angular vel

        a2g, a_origin_in_g = CoordinateFrame.transform_from_a_to_b(frameA, self)
        b2g, b_origin_in_g = CoordinateFrame.transform_from_a_to_b(frameB, self)

        new_pos = a_origin_in_g - b_origin_in_g
        new_rot = a2g * b2g.inv()  # self -> new frame

        return new_rot, new_pos

    # returns a relative frame from two global uninitialized frames
    @staticmethod
    def relative_frame_a_to_b(a2g, a_origin_in_g, b2g, b_origin_in_g):
        # a -> g -> b
        a2b = b2g.inv() * a2g
        # delta vector is rotated from g to a frame
        a_origin_in_b = b2g.apply(a_origin_in_g - b_origin_in_g, inverse=True)
        return a2b, a_origin_in_b

    @staticmethod
    def pose_a_view_in_b(poseA: np.ndarray, frameA, frameB, mode="xyz"):
        # assumes pose A is in A frame
        p_in_a = poseA[..., :3]
        p2a = R.from_euler(mode, poseA[..., 3:])
        a2b, a_in_b = CoordinateFrame.transform_from_a_to_b(frameA, frameB)
        p2b = a2b * p2a
        p_in_b = a2b.apply(p_in_a) + a_in_b
        return np.concatenate([p_in_b, p2b.as_euler(mode)], axis=-1)

    # takes a pose in (self) frame and transforms it by the A->B transform
    def pose_apply_a_to_b(self, pose: np.ndarray, frameA, frameB, mode="xyz"):
        # a -> g -> b
        pa_in_self = pose[..., :3]
        pa2self = R.from_euler(mode, pose[..., 3:])
        a2b, a_in_b = CoordinateFrame.transform_from_a_to_b(frameA, frameB)

        pb2self = pa2self * a2b.inv()
        pb_in_self = pa_in_self + pb2self.apply(-a_in_b)  # shift a to b (relative to self. frame)

        return np.concatenate([pb_in_self, pb2self.as_euler(mode)], axis=-1)

    @staticmethod
    def from_pose(pose, parent_frame, mode="xyz"):
        position = pose[:3]
        eul = pose[3:]
        r = R.from_euler(mode, eul)
        return CoordinateFrame(parent_frame, r.inv(), position)

    def as_pose(self, base_frame, mode="xyz"):
        self2b, self_in_b = CoordinateFrame.transform_from_a_to_b(self, base_frame)
        return np.concatenate([self_in_b, self2b.as_euler(mode)])

    @property
    def pos(self) -> np.ndarray:
        return self.c_origin_in_g

    @property
    def orn(self) -> np.ndarray:
        return self.c2g_R.as_quat()

    @property
    def rot(self) -> R:
        return self.c2g_R

    def __eq__(self, other):
        return np.allclose(self.c2g_R.as_matrix(), other.c2g_R.as_matrix()) and np.allclose(self.c_origin_in_g, other.c_origin_in_g)

    def __str__(self):
        return "CoordinateFrame - c_in_g: %s | c2g(eul): %s" % (self.c_origin_in_g, self.c2g_R.as_euler("xyz"))

    def __copy__(self):
        return CoordinateFrame(self.parent, self.c2p_R.inv(), self.c_origin_in_p)


# defined as x right, y forward, z up
world_frame_3D = CoordinateFrame(parent=None)


## PYMESH STUFF

def pymesh_translate(mesh, translation_vec):
    return pymesh.form_mesh(mesh.vertices + np.asarray(translation_vec)[None], mesh.faces)


def pymesh_rotate(mesh, rotation: R):
    return pymesh.form_mesh(rotation.apply(mesh.vertices), mesh.faces)


def pymesh_rotate_and_translate(mesh, rotation: R, translation_vec: np.ndarray, trans_first=False):
    verts = mesh.vertices
    if trans_first:
        verts = verts + translation_vec[None]
    verts = rotation.apply(verts)
    if not trans_first:
        verts = verts + translation_vec[None]

    return pymesh.form_mesh(verts, mesh.faces)


def pymesh_transform_a_to_b(mesh, a: CoordinateFrame, b: CoordinateFrame):
    a2b, a_in_b = CoordinateFrame.transform_from_a_to_b(a, b)
    return pymesh_rotate_and_translate(mesh, a2b, a_in_b)


def pymesh_project_mesh_points_onto_plane(raw_mesh, frame, parent_frame, plane_normal, plane_center):
    vertices = np.asarray(raw_mesh.vertices)
    vertices = CoordinateFrame.point_from_a_to_b(vertices, frame, parent_frame)
    return np_project_points_onto_plane(vertices, plane_normal, plane_center)


def np_project_points_onto_plane(pts, plane_normal, plane_center):
    n = np.asarray(plane_normal)
    n /= np.linalg.norm(n)  # enforce its normal
    c = np.asarray(plane_center)

    projection = pts - (pts - c).dot(n)[..., None] * n[None]

    return projection


def pymesh_gen_ellipse_cylinder(rx0, ry0, rx1, ry1, p0, p1, num_segments):
    assert (len(p0) == 3)
    assert (len(p1) == 3)
    Z = np.array([0, 0, 1], dtype=float)
    p0 = np.array(p0, dtype=float)
    p1 = np.array(p1, dtype=float)
    axis = p1 - p0
    l = np.linalg.norm(axis)
    if l <= 1e-12:
        axis = Z

    angles = [2 * np.pi * i / float(num_segments) for i in range(num_segments)]
    brim = np.array([[rx0 / ry0 * np.cos(theta), np.sin(theta), 0.0]
                     for theta in angles])
    trim = np.array([[rx1 / ry1 * np.cos(theta), np.sin(theta), 0.0]
                     for theta in angles])
    rot = pymesh.Quaternion.fromData(Z, axis).to_matrix()

    bottom_rim = np.dot(rot, brim.T).T * ry0 + p0
    top_rim = np.dot(rot, trim.T).T * ry1 + p1

    vertices = np.vstack([[p0, p1], bottom_rim, top_rim])

    bottom_fan = np.array([
        [0, (i + 1) % num_segments + 2, i + 2]
        for i in range(num_segments)], dtype=int)

    top_fan = np.array([
        [1, i + num_segments + 2, (i + 1) % num_segments + num_segments + 2]
        for i in range(num_segments)], dtype=int)

    side = np.array([
        [[2 + i, 2 + (i + 1) % num_segments, 2 + i + num_segments],
         [2 + i + num_segments, 2 + (i + 1) % num_segments, 2 + (i + 1) % num_segments + num_segments]]
        for i in range(num_segments)], dtype=int)
    side = side.reshape((-1, 3), order="C")

    faces = np.vstack([bottom_fan, top_fan, side])
    return pymesh.form_mesh(vertices, faces), rot


def pymesh_gen_ellipse_tube(rx0_out, ry0_out, rx1_out, ry1_out, rx0_in, ry0_in, rx1_in, ry1_in, p0, p1, num_segments,
                            with_quad=False):
    assert (len(p0) == 3)
    assert (len(p1) == 3)
    Z = np.array([0, 0, 1], dtype=float)
    p0 = np.array(p0, dtype=float)
    p1 = np.array(p1, dtype=float)
    axis = p1 - p0
    l = np.linalg.norm(axis)
    if l <= 1e-12:
        axis = Z
    N = num_segments

    angles = [2 * np.pi * i / float(N) for i in range(N)]
    brim_out = np.array([[rx0_out / ry0_out * np.cos(theta), np.sin(theta), 0.0]
                         for theta in angles])
    brim_in = np.array([[rx0_in / ry0_in * np.cos(theta), np.sin(theta), 0.0]
                        for theta in angles])
    trim_out = np.array([[rx1_out / ry1_out * np.cos(theta), np.sin(theta), 0.0]
                         for theta in angles])
    trim_in = np.array([[rx1_in / ry1_in * np.cos(theta), np.sin(theta), 0.0]
                        for theta in angles])
    rot = pymesh.Quaternion.fromData(Z, axis).to_matrix()

    bottom_outer_rim = np.dot(rot, brim_out.T).T * ry0_out + p0
    bottom_inner_rim = np.dot(rot, brim_in.T).T * ry0_in + p0
    top_outer_rim = np.dot(rot, trim_out.T).T * ry1_out + p1
    top_inner_rim = np.dot(rot, trim_in.T).T * ry1_in + p1

    vertices = np.vstack([
        bottom_outer_rim,
        bottom_inner_rim,
        top_outer_rim,
        top_inner_rim])

    if with_quad:
        top = np.array([
            [2 * N + i, 2 * N + (i + 1) % N, 3 * N + (i + 1) % N, 3 * N + i]
            for i in range(N)])
        bottom = np.array([
            [(i + 1) % N, i, N + i, N + (i + 1) % N]
            for i in range(N)])
        inner = np.array([
            [3 * N + i, 3 * N + (i + 1) % N, N + (i + 1) % N, N + i]
            for i in range(N)])
        outer = np.array([
            [i, (i + 1) % N, 2 * N + (i + 1) % N, 2 * N + i]
            for i in range(N)])
        faces = np.vstack([top, bottom, inner, outer])
    else:
        top = np.array([
            [[2 * N + i, 2 * N + (i + 1) % N, 3 * N + i],
             [3 * N + i, 2 * N + (i + 1) % N, 3 * N + (i + 1) % N]
             ] for i in range(N)])
        bottom = np.array([
            [[(i + 1) % N, i, N + i],
             [(i + 1) % N, N + i, N + (i + 1) % N]
             ] for i in range(N)])
        inner = np.array([
            [[3 * N + i, 3 * N + (i + 1) % N, N + i],
             [N + i, 3 * N + (i + 1) % N, N + (i + 1) % N]
             ] for i in range(N)])
        outer = np.array([
            [[i, (i + 1) % N, 2 * N + i],
             [2 * N + i, (i + 1) % N, 2 * N + (i + 1) % N]
             ] for i in range(N)])

        faces = np.vstack([
            top.reshape((-1, 3)),
            bottom.reshape((-1, 3)),
            inner.reshape((-1, 3)),
            outer.reshape((-1, 3))])
    return pymesh.form_mesh(vertices, faces), rot


# returns insidemesh, outsidemesh.. direction specified by in_direction and normal
# bounded bc with the in direction, we can come up with an upper bound on the slice (AKA the returned geometry is a superset of the true slice)
def pymesh_slice_ellipse_cylinder_approximate_bounded(p0, p1, base_rot, frame, base_frame, rx0, ry0, rx1, ry1, plane_normal, plane_center, num_segments, in_direction=1):
    a = p1 - p0

    # normal points INTO shape
    n = np.asarray(plane_normal) * in_direction

    v = (a.dot(n) * a) - (a.dot(a) * n)
    v_raw = base_rot.T @ CoordinateFrame.point_from_a_to_b(v, base_frame, frame)
    theta = np.arctan2(rx0 * v_raw[1], ry0 * v_raw[0])
    radius0 = np.linalg.norm([rx0 * np.cos(theta), ry0 * np.sin(theta)])
    radius1 = np.linalg.norm([rx1 * np.cos(theta), ry1 * np.sin(theta)])  # radii in the plane formed by a and n

    vn = v / np.linalg.norm(v)
    if vn.dot(n) < 0:
        vn = -vn  # flip to face towards in direction

    p0_in_edge = p0 + vn * radius0
    p1_in_edge = p1 + vn * radius1

    # NOW, where does (p1_in_edge - p0_in_edge) intersect with the plane??
    a_edge = p1_in_edge - p0_in_edge

    p0_delta_dir = (p0_in_edge - plane_center).dot(n)
    t = - p0_delta_dir / a_edge.dot(n)

    if 0 <= t <= 1:
        rxt = rx0 + t*(rx1 - rx0)
        ryt = ry0 + t*(ry1 - ry0)
        pt = p0 + t * a  # use t from before, but along radial axis
        p0_side_mesh, _ = pymesh_gen_ellipse_cylinder(rx0, ry0, rxt, ryt, p0, pt, num_segments)
        p1_side_mesh, _ = pymesh_gen_ellipse_cylinder(rxt, ryt, rx1, ry1, pt, p1, num_segments)
        if p0_delta_dir > 0:
            return p0_side_mesh, p1_side_mesh
        else:
            return p1_side_mesh, p0_side_mesh

    # cylinder doesn't intersect plane
    return None


def batch_points_within_ellipse_2d(arr: np.ndarray, center: np.ndarray, radii: np.ndarray):
    """
    arr is (...., 2), assumes x,y are axes of ellipse (do transform beforehand if not)
    
    returns if (arr.x-h)^2/a^2 + (arr.y-k)^2/b^2 <= 1
    """

    # (...., ) boolean array
    return np.divide(np.square(arr - center), np.square(radii)).sum(-1) <= 1


def batch_points_within_elliptical_tube_3d(arr: np.ndarray, center: np.ndarray, radii: np.ndarray, depth,
                                           allow_past_depth=True):
    """
    arr is (...., 3), assumes x,y, are axes of ellipse (do transform beforehand if not), z is depth

    tube is centered
    returns if (arr.x-h)^2/a^2 + (arr.y-k)^2/b^2 <= 1   and  |arr.z| < depth / 2
      or if (allow_past_depth), returns if " ^^ "       or  |arr.z| > depth / 2

    """

    # (...., ) boolean array
    xy = arr[..., :2]
    z = arr[..., 2]

    within_2d = batch_points_within_ellipse_2d(xy, center[:2], radii)

    if allow_past_depth:
        return np.logical_or(within_2d, np.abs(z) > depth / 2.)  # allow being past the tube bounds
    else:
        return np.logical_and(within_2d, np.abs(z) < depth / 2.)

def ee_true_frame_from_state(position, orientation, world_frame):
    return CoordinateFrame(world_frame, R.from_quat(orientation).inv(), position)  # true ee_frame


# applying rotations
def torch_quat_mul(q1, q2, real_last=True):
    q1, q2 = torch.broadcast_tensors(q1, q2)
    if real_last:
        b, c, d, a = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        f, g, h, e = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    else:
        a, b, c, d = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        e, f, g, h = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    _a = a*e - b*f - c*g - d*h
    _b = b*e + a*f + c*h - d*g
    _c = a*g - b*g + c*e + d*f
    _d = a*h + b*g - c*f + d*e

    return torch_quat(_a, _b, _c, _d, real_last=real_last)


def torch_quat_angle(q1, q2, approximate=False, real_last=True):
    # https://math.stackexchange.com/questions/90081/quaternion-distance
    q1, q2 = torch.broadcast_tensors(q1, q2)
    _el_mul = (q1 * q2)
    # if real_last:
    #     real, img = _el_mul[..., 3], _el_mul[..., :3]
    # else:
    #     real, img = _el_mul[..., 0], _el_mul[..., 1:4]
    # dot = real - img.sum(-1)
    # arccos(real) gives you the half angle of rotation
    dot = _el_mul.sum(-1)
    # set tol high enough to prevent > 1 (fp-precision bugs)
    # dot = torch.where(dot > 0, torch.clamp_min(dot - tol, 0), torch.clamp_max(dot + tol, 0))
    # trick to avoid floating point errors (and therefore NaNs)
    # dot_sq = torch.clamp(2 * (dot ** 2) - 1, min=-1 + tol, max=1 - tol)
    if approximate:
        return 1 - dot ** 2
    else:
        # this can yield NaN gradients?
        cond = dot.abs() > 1
        dot[cond] /= dot[cond].abs().detach()
        dot_sq = torch.clamp(2 * (dot ** 2) - 1, min=-1, max=1)
        return torch.arccos(dot_sq)


def torch_rpt_angle(rpt1, rpt2, br=0, bp=-np.pi/2, bt=0, approximate=False):
    q1 = torch_rpt_to_quat(rpt1, br=br, bp=bp, bt=bt)
    # rpt1.retain_grad()
    # q1.retain_grad()
    # loss = (q1 ** 2).sum()
    # loss.backward()
    # print("rpt1", rpt1.grad.isfinite().all())
    # print("q1", q1.grad.isfinite().all())
    q2 = torch_rpt_to_quat(rpt2, br=br, bp=bp, bt=bt)
    return torch_quat_angle(q1, q2, approximate=approximate)


def torch_eul_angle(eul1, eul2, approximate=False):
    q1 = torch_eul_to_quat(eul1)
    q2 = torch_eul_to_quat(eul2)
    return torch_quat_angle(q1, q2, approximate=approximate)


def torch_eul_to_quat(eul, real_last=True):
    assert eul.shape[-1] == 3
    x, y, z = eul[..., 0], eul[..., 1], eul[..., 2]
    cx = torch.cos(x * 0.5)
    sx = torch.sin(x * 0.5)
    cy = torch.cos(y * 0.5)
    sy = torch.sin(y * 0.5)
    cz = torch.cos(z * 0.5)
    sz = torch.sin(z * 0.5)

    _a = cx * cy * cz + sx * sy * sz
    _b = sx * cy * cz - cx * sy * sz
    _c = cx * sy * cz + sx * cy * sz
    _d = cx * cy * sz - sx * sy * cz
    return torch_quat(_a, _b, _c, _d, real_last=real_last)


def torch_rpt_to_quat(rpt, br=0, bp=-np.pi/2, bt=0, real_last=True):
    assert rpt.shape[-1] == 3
    x, y, z = rpt[..., 0] + br, rpt[..., 1] + bp, rpt[..., 2] + bt
    cx = torch.cos(x * 0.5)
    sx = torch.sin(x * 0.5)
    cy = torch.cos(y * 0.5)
    sy = torch.sin(y * 0.5)
    cz = torch.cos(z * 0.5)
    sz = torch.sin(z * 0.5)

    cxcz, sxsz = cx * cz, sx * sz
    cxsz, sxcz = cx * sz, sx * cz

    _a = cy * (cxcz - sxsz)
    _b = sy * (cxcz + sxsz)
    _c = sy * (cxsz - sxcz)
    _d = cy * (cxsz + sxcz)
    return torch_quat(_a, _b, _c, _d, real_last=real_last)


def torch_quat(a, b, c, d, real_last=True):
    if real_last:
        return torch.stack([b, c, d, a], dim=-1)
    else:
        return torch.stack([a, b, c, d], dim=-1)


# applying rotations
def np_quat_mul(q1, q2, real_last=True):
    q1, q2 = np.broadcast_arrays(q1, q2)
    if real_last:
        b, c, d, a = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        f, g, h, e = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    else:
        a, b, c, d = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        e, f, g, h = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    _a = a*e - b*f - c*g - d*h
    _b = b*e + a*f + c*h - d*g
    _c = a*g - b*g + c*e + d*f
    _d = a*h + b*g - c*f + d*e

    return np_quat(_a, _b, _c, _d, real_last=real_last)


def np_quat_angle(q1, q2, real_last=True):
    # https://math.stackexchange.com/questions/90081/quaternion-distance
    q1, q2 = np.broadcast_arrays(q1, q2)
    _el_mul = (q1 * q2)
    # if real_last:
    #     real, img = _el_mul[..., 3], _el_mul[..., :3]
    # else:
    #     real, img = _el_mul[..., 0], _el_mul[..., 1:4]
    # dot = real - img.sum(-1)
    # arccos(real) gives you the half angle of rotation
    dot = _el_mul.sum(-1)
    # set tol high enough to prevent > 1 (fp-precision bugs)
    # dot = np.where(dot > 0, torch.clamp_min(dot - tol, 0), np.clamp_max(dot + tol, 0))
    cond = np.abs(dot) > 1
    div = np.divide(dot, np.abs(dot), where=cond)
    dot = np.where(cond, div, dot)
    dot_sq = np.clip(2 * (dot ** 2) - 1, a_min=-1, a_max=1)
    return np.arccos(dot_sq)


def np_rpt_angle(rpt1, rpt2, br=0, bp=-np.pi/2, bt=0):
    q1 = np_rpt_to_quat(rpt1, br=br, bp=bp, bt=bt)
    q2 = np_rpt_to_quat(rpt2, br=br, bp=bp, bt=bt)
    return np_quat_angle(q1, q2)


def np_eul_angle(eul1, eul2, br=0, bp=-np.pi/2, bt=0):
    q1 = np_eul_to_quat(eul1)
    q2 = np_eul_to_quat(eul2)
    return np_quat_angle(q1, q2)


def np_eul_to_quat(eul, real_last=True):
    assert eul.shape[-1] == 3
    x, y, z = eul[..., 0], eul[..., 1], eul[..., 2]
    cx = np.cos(x * 0.5)
    sx = np.sin(x * 0.5)
    cy = np.cos(y * 0.5)
    sy = np.sin(y * 0.5)
    cz = np.cos(z * 0.5)
    sz = np.sin(z * 0.5)

    _a = cx * cy * cz + sx * sy * sz
    _b = sx * cy * cz - cx * sy * sz
    _c = cx * sy * cz + sx * cy * sz
    _d = cx * cy * sz - sx * sy * cz
    return np_quat(_a, _b, _c, _d, real_last=real_last)


def np_rpt_to_quat(rpt, br=0, bp=-np.pi/2, bt=0, real_last=True):
    assert rpt.shape[-1] == 3
    x, y, z = rpt[..., 0] + br, rpt[..., 1] + bp, rpt[..., 2] + bt
    cx = np.cos(x * 0.5)
    sx = np.sin(x * 0.5)
    cy = np.cos(y * 0.5)
    sy = np.sin(y * 0.5)
    cz = np.cos(z * 0.5)
    sz = np.sin(z * 0.5)

    cxcz, sxsz = cx * cz, sx * sz
    cxsz, sxcz = cx * sz, sx * cz

    _a = cy * (cxcz - sxsz)
    _b = sy * (cxcz + sxsz)
    _c = sy * (cxsz - sxcz)
    _d = cy * (cxsz + sxcz)
    return np_quat(_a, _b, _c, _d, real_last=real_last)


def np_quat(a, b, c, d, real_last=True):
    if real_last:
        return np.stack([b, c, d, a], axis=-1)
    else:
        return np.stack([a, b, c, d], axis=-1)


# @numba.jit(forceobj=True, cache=True)
def batch_euler_to_matrix(eul: np.ndarray):
    base = list(eul.shape[:-1])
    assert eul.shape[-1] == 3, eul.shape
    # eul = eul % (2 * np.pi)
    cp = np.cos(eul[..., 0])  # pitch, phi
    sp = np.sin(eul[..., 0])
    cy = np.cos(eul[..., 1])  # yaw, theta
    sy = np.sin(eul[..., 1])
    cr = np.cos(eul[..., 2])  # roll, psi
    sr = np.sin(eul[..., 2])

    # slight optimization
    spsy = sp * sy
    cpsy = cp * sy

    flat = [cy * cr,    -cp * sr + spsy * cr,   sp * sr + cpsy * cr,
            cy * sr,    cp * cr + spsy * sr,    -sp * cr + cpsy * sr,
            -sy,        sp * cy,                cp * cy]

    # flat = [cy * cr + sy * sp * sr, cp * sr,            sr * cy * sp - sy * cr,
    #         cr * sy * sp - sr * cy, cr * cp,            sy * sr + cr * cy * sp,
    #         cp * sy,                -sp,                cp * cy]
    # flat = [cr * cp - cy * sr * sp,     - sr * cp - cy * sp * cr,   sy * sp,
    #         cr * sp + cy * cp * sr,     - sp * sr + cy * cp * cr,   -sy * cp,
    #         sy * sr,                    sy * cr,                    cy]
    # ... x 9

    return np.stack(flat, axis=-1).reshape(base + [3, 3])


def fast_pose_grab_diff_fn(a0, b0):
    a = to_numpy(a0, check=True)
    b = to_numpy(b0, check=True)
    pos_diff = a[..., :3] - b[..., :3]
    mat_a = batch_euler_to_matrix(a[..., 3:6])
    mat_b = batch_euler_to_matrix(b[..., 3:6])
    # mat_a = R.from_euler("xyz", a[..., 3:6]).as_matrix()
    # mat_b = R.from_euler("xyz", b[..., 3:6]).as_matrix()
    orn_diff = batch_orientation_error(mat_a,
                                       mat_b)
    grab_diff = a[..., -1:] - b[..., -1:]
    out = np.concatenate([pos_diff, orn_diff, grab_diff], axis=-1)
    return to_torch(out, device=a0.device) if isinstance(a0, torch.Tensor) else out


def fast_pose_diff_fn(a0, b0):
    a = to_numpy(a0, check=True)
    b = to_numpy(b0, check=True)
    pos_diff = a[..., :3] - b[..., :3]
    mat_a = batch_euler_to_matrix(a[..., 3:6])
    mat_b = batch_euler_to_matrix(b[..., 3:6])
    # mat_a = R.from_euler("xyz", a[..., 3:6]).as_matrix()
    # mat_b = R.from_euler("xyz", b[..., 3:6]).as_matrix()
    orn_diff = batch_orientation_error(mat_a,
                                       mat_b)
    out = np.concatenate([pos_diff, orn_diff], axis=-1)
    return to_torch(out, device=a0.device) if isinstance(a0, torch.Tensor) else out


def clip_ee_orientation_conical(ori_eul, ee_axis, world_axis, max_theta):
    """ Clips ori_eul such that ee_axis represented in world frame fits inside conical region around world axis.
    Rotation from Vectors A -> B: https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d

    :param ori_eul: euler angles representing rotation of ee2world.
    :param ee_axis: 3D, unit
    :param world_axis: 3D, unit
    :param max_theta: cone angle
    :return:
    """

    ee_axis /= np.linalg.norm(ee_axis)
    world_axis /= np.linalg.norm(world_axis)
    ee2w = T.euler2mat(ori_eul)
    ee_axis_in_w = ee2w @ ee_axis

    # angle between the two.
    angle = np.arccos(np.dot(ee_axis_in_w, world_axis))

    if angle > max_theta:
        c = np.cos(angle - max_theta)
        s_old = np.sin(angle)  # original
        if c > -1 + 1e-11 and s_old > 0:
            # outside of cone, and angle / angle-max_angle are not 180 degrees off from world_axis
            v = np.cross(ee_axis_in_w, world_axis)  # rotate the ee axis to the world.
            v /= np.linalg.norm(v)  # axis
            d_angle = angle - max_theta

            # logger.debug(f"angle: {d_angle}, axis: {v}")
            eeaxis2clippedaxis = R.from_rotvec(v * d_angle).as_matrix()
            return T.mat2euler(eeaxis2clippedaxis @ ee2w)  # correction in world frame
        else:
            return ori_eul  # unsure what to do here, but should never happen.
    else:
        return ori_eul  # no change, within cone



if __name__ == '__main__':
    from sbrl.utils import math_utils

    # parent_to_child_rot always defines the rotation to go from points in parent frame to their equivalent point in child frame
    # NOTE: this is not the transform that should be applied to the parent coordinate axis to get the child axis

    # translation +1 in x
    frame1 = CoordinateFrame(world_frame_3D, R.from_matrix(np.eye(3)), child_origin=np.array([1.0, 0, 0]))

    # child: x is global -y, y is global x
    vec = np.array([0, 0, 1.0])
    theta = np.pi / 2  # xy plane (cc)
    frame2 = CoordinateFrame(frame1, R.from_rotvec(theta * vec), child_origin=np.array([0.0, 1.0, 0.0]))

    # y up, z in the global -y
    vec = np.array([1.0, 0.0, 0.0])
    theta = - np.pi / 2
    frame3 = CoordinateFrame(world_frame_3D, R.from_rotvec(vec * theta), child_origin=np.array([0, 0.0, 1.0]))

    # compared to frame3, z in the x direction, x in the -z direction
    # global: z in the x direction, x in the y direction, y up
    vec = np.array([0.0, 1.0, 0.0])
    theta = np.pi / 2
    frame4 = CoordinateFrame(frame3, R.from_rotvec(vec * theta), child_origin=np.array([0.0, 0.0, -1.0]))

    # 1->2 on 4
    frame5 = frame4.apply_a_to_b(frame1, frame2)

    # to summarize
    #  world  -->  frame1(no rot) --> frame2
    #        |
    #         -->  frame3 --> frame4 --> frame5 (1->2 on 4)

    ## GLOBALS

    f1_basis = np.column_stack(world_frame_3D.get_basis_vectors_of_frame(frame1))
    f2_basis = np.column_stack(world_frame_3D.get_basis_vectors_of_frame(frame2))
    f3_basis = np.column_stack(world_frame_3D.get_basis_vectors_of_frame(frame3))
    f4_basis = np.column_stack(world_frame_3D.get_basis_vectors_of_frame(frame4))
    f5_basis = np.column_stack(world_frame_3D.get_basis_vectors_of_frame(frame5))

    assert np.allclose(f1_basis, np.eye(3)), f1_basis
    assert np.allclose(f2_basis, np.array([[0., 1., 0], [-1., 0., 0], [0, 0, 1.]])), f2_basis
    assert np.allclose(f3_basis, np.array([[1., 0., 0], [0., 0., -1], [0, 1, 0.]])), f3_basis
    assert np.allclose(f4_basis, np.array([[0., 0., -1.], [-1., 0., 0.], [0, 1, 0.]])), f4_basis
    assert np.allclose(f5_basis, np.array([[0., 0., -1.], [0., -1., 0.], [-1., 0., 0.]])), f5_basis

    equivalent_points = [
        [[0, 0, 0], [1., 0, 0], [1., -1, 0], [1, -1, -1]],
        [[0, 1, 0], [0., 0, 0], [1., -1, -1.], [0, -1, -1]],
    ]

    for point_set in equivalent_points:
        p1, p2, p3, p4 = [np.asarray(p) for p in point_set]

        assert np.allclose(p2, CoordinateFrame.point_from_a_to_b(p1, frame1, frame2))
        assert np.allclose(p3, CoordinateFrame.point_from_a_to_b(p1, frame1, frame3))
        assert np.allclose(p4, CoordinateFrame.point_from_a_to_b(p1, frame1, frame4))
        assert np.allclose(p3, CoordinateFrame.point_from_a_to_b(p2, frame2, frame3))
        assert np.allclose(p4, CoordinateFrame.point_from_a_to_b(p2, frame2, frame4))
        assert np.allclose(p4, CoordinateFrame.point_from_a_to_b(p3, frame3, frame4))

    # transform locals
    # f3 -> f4 is the same as f1 to f2 in the global frame
    f3_f4, f3_m_f4_world = world_frame_3D.transform_from_a_to_b_local(frame3, frame4)
    f1_f2, f1_m_f2_world = world_frame_3D.transform_from_a_to_b_local(frame1, frame2)

    assert np.allclose(f3_f4.as_matrix(), f1_f2.as_matrix())
    assert np.allclose(f3_m_f4_world, f1_m_f2_world)

    # pose stuff
    assert frame1 == CoordinateFrame.from_pose(frame1.as_pose(world_frame_3D), world_frame_3D)
    p1 = frame1.as_pose(world_frame_3D)

    assert np.allclose(CoordinateFrame.pose_a_view_in_b(p1, world_frame_3D, frame1), np.zeros(6))

    assert CoordinateFrame.from_pose(CoordinateFrame.pose_a_view_in_b(p1, world_frame_3D, frame2), frame2) == frame1

    assert CoordinateFrame.from_pose(world_frame_3D.pose_apply_a_to_b(frame1.as_pose(world_frame_3D), frame1, frame2), world_frame_3D) == frame2

    #### MAZE ####

    # Maze dimensions (ncols, nrows)
    nx, ny = 4, 4
    # Maze entry position
    ix, iy = 0, 0

    maze = Maze(nx, ny, ix, iy)
    maze.make_maze()

    new_maze = Maze(nx, ny, ix, iy)
    new_maze.from_numpy(maze.to_numpy())

    mod_maze = Maze(nx, ny, ix, iy)
    npmod_maze = mod_maze.to_numpy()
    npmod_maze[0, 0] = 15 - npmod_maze[0, 0]
    mod_maze.from_numpy(npmod_maze)

    assert maze == new_maze
    assert maze != mod_maze

    # print(maze)
    # maze.write_svg("test.svg")

    #### ROTATIONS ####

    eul = np.random.uniform(0., np.pi, size=(1, 3))

    # EUL -> MAT
    mat = batch_euler_to_matrix(eul)
    mat_true = R.from_euler("xyz", eul).as_matrix()
    eul_rev = R.from_matrix(mat).as_euler("xyz")

    assert (np.linalg.norm(mat - mat_true, axis=(-1, -2), ord='fro') < 1e-11).all(), [mat, mat_true]

    eul1 = np.random.uniform(0., np.pi, size=(10, 3))
    eul2 = np.random.uniform(0., np.pi, size=(10, 3))
    r1 = R.from_euler("xyz", eul1)
    r2 = R.from_euler("xyz", eul2)

    # EUL -> QUAT
    q = np_eul_to_quat(eul1)

    mat = R.from_quat(q).as_matrix()
    mat_true = r1.as_matrix()
    assert (np.linalg.norm(mat - mat_true, axis=(-1, -2), ord='fro') < 1e-11).all(), [mat, mat_true]

    # RPT -> QUAT
    q = np_rpt_to_quat(math_utils.convert_eul_to_rpt(eul1))

    mat = R.from_quat(q).as_matrix()
    assert (np.linalg.norm(mat - mat_true, axis=(-1, -2), ord='fro') < 1e-11).all(), [mat, mat_true]

    # QUAT DISTANCE
    eul1 = np.random.uniform(0., np.pi, size=(10, 3))
    eul2 = eul1.copy()
    eul2[:, 2] += np.pi
    d11 = np_eul_angle(eul1, eul1)
    d12 = np_eul_angle(eul1, eul2)
    d11_t = torch_eul_angle(to_torch(eul1, device="cpu"), to_torch(eul1, device="cpu"))
    assert np.all(np.abs(d11) < 1e-5), [d11]
    assert (torch.abs(d11_t) < 1e-5).all(), [d11_t]
    assert np.all(np.abs(d12 - np.pi) < 1e-5), [d12]
