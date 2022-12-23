import numpy as np
from sbrl.sandbox.bt_rrt_helpers import get_bt_spatial_cost_fn

from sbrl.utils.python_utils import AttrDict

# low = np.array([-.2, -.2, -.2])
low = np.array([-.15, -.15, 0.03])
high = -low
high[0] = 0.  # y axis
high[1] = 0.  # y axis
high[2] = 0.5  # z axis

# data = np.random.uniform(low, high, (1000, 3))

ranges = np.linspace(low, high, 50)
range_z = np.linspace(low[2], high[2], 50)
print(ranges)
x, y, z = np.meshgrid(*np.split(ranges[:, :2], 2, axis=-1), range_z)
cost_fn = get_bt_spatial_cost_fn(AttrDict())

# x,y,z,v are all (N x N x N)
v = cost_fn(x, y, z)

separation = 0.01
val = 0.5

shape_inside = np.where(v < val)
x = x
y = y
z = z
v = v

# outside_cyl = np.sqrt(x ** 2 + y ** 2) > 0.25
# v[outside_cyl] = 100.

import mayavi.mlab as mlab

print(x.max(), x.min(), x.shape)
print(y.max(), y.min(), y.shape)
print(z.max(), z.min(), z.shape)
# contours=[0.7, 0.5, 0.3, 0.2, 0.1]
mlab.contour3d(v, colormap="YlOrRd", contours=[0.7, 0.5, 0.3, 0.2, 0.1], extent=[x.min(), x.max(), y.min(), y.max(), z.min(), z.max()], opacity=1., transparent=False, vmin=0.0, vmax=0.7)

# def test_contour3d():
#     x, y, z = np.ogrid[-5:5:64j, -5:5:64j, -5:5:64j]
#
#     scalars = x * x * 0.5 + y * y + z * z * 2.0
#
#     obj = mlab.contour3d(scalars, contours=4, transparent=True)
#     return obj
#
# test_contour3d()
mlab.show()

#
# x,y,z = np.split(data, 3, axis=-1)
# v = cost_fn(x, y, z)
# cmax = cost_fn(np.ones(1), np.ones(1), np.zeros(1))
#
# print(np.min(v))
# print(np.max(v))
# c = 1 - v
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# cmhot = plt.get_cmap("hot")
# ax.scatter(x, y, z, s=15, c=c.flatten(), cmap=cmhot)
#
# if args.save_file is not None:
#     logger.debug("Saving to file -> %s" % args.save_file)
#     fig.savefig(args.save_file)

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# # fig, ax = plt.subplots()
# # cs = plt.contourf(x, y, np.zeros_like(x), levels=np.linspace(0, 1., 10), cmap="hot")
# # fig.colorbar(cs)
# def draw_contour(frame):
#     z = ranges[frame, 2]
#     same_z = z * np.ones_like(x)
#     v = cost_fn(x, y, same_z)  # then black -> yellow/white(best)
#
#     where_in_bounds = val - separation < (v < val + separation)
#     idxs = where_in_bounds.nonzero()
#
#     # print(same_z.shape, x.shape)
#     z = z * np.ones_like(x[idxs])
#     print(idxs)
#
#     ax.plot_surface(x[where_in_bounds], y[where_in_bounds], z)
#     # ax.clear()
#     # cf = ax.contour3D(x, y, v, 10, cmap="hot")
#     #                  # levels=np.linspace(0, 1., 10), vmin=0., vmax=1.)
#     # cf.changed()
#
#     # mouth_radii = (0.025, 0.02)
#     # theta = np.linspace(0, np.pi * 2, 100)
#     # ax.plot(mouth_radii[0] * np.cos(theta), mouth_radii[1] * np.sin(theta), color="black")
#
#     ax.set_title("Cost for distance from mouth: z = %.2f cm" % (z * 100))
#     ax.set_xlabel("right(+) / left (m)")
#     ax.set_ylabel("up(+) / down (m)")
#
# # ani = FuncAnimation(fig, draw_contour, frames=100, interval=100, blit=False, repeat=False)
#
# draw_contour(0)
#
# plt.show()
#
# # ani.save("test.mp4")
