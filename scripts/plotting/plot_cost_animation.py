import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from sbrl.sandbox.bt_rrt_helpers import get_bt_spatial_cost_fn

from sbrl.utils.python_utils import AttrDict

low = np.array([-.2, -.2, 1e-5])
high = -low
high[2] = 0.2  # z axis

# data = np.random.uniform(low, high, (1000, 3))

ranges = np.linspace(low, high, 100)
print(ranges)
x,y = np.meshgrid(*np.split(ranges[:, :2], 2, axis=-1))
cost_fn = get_bt_spatial_cost_fn(AttrDict())
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

fig, ax = plt.subplots()
cs = plt.contourf(x, y, np.zeros_like(x), levels=np.linspace(0, 1., 10), cmap="hot")
fig.colorbar(cs)
def animate(frame):
    z = ranges[frame % 100, 2]
    same_z = z * np.ones_like(x)
    v = cost_fn(x, y, same_z)  # then black -> yellow/white(best)
    ax.clear()
    cf = ax.contourf(x, y,  v.reshape(x.shape), cmap="hot",
                     levels=np.linspace(0, 1., 10), vmin=0., vmax=1.)
    cf.changed()

    mouth_radii = (0.025, 0.02)
    theta = np.linspace(0, np.pi * 2, 100)
    ax.plot(mouth_radii[0] * np.cos(theta), mouth_radii[1] * np.sin(theta), color="black")

    ax.set_title("Cost for distance from mouth: z = %.2f cm" % (z * 100))
    ax.set_xlabel("right(+) / left (m)")
    ax.set_ylabel("up(+) / down (m)")

ani = FuncAnimation(fig, animate, interval=100, blit=False, repeat=False)


plt.show()

# ani.save("test.mp4")
