from multiprocessing import Pipe, Process

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

from sbrl.experiments import logger
from sbrl.utils.core_utils import CloudPickleWrapper
from sbrl.utils.python_utils import AttrDict, get_with_default
from sbrl.utils.torch_utils import pad_dims

orange = np.array([255, 145, 50, 255]) / 255
teal = np.array([5, 152, 176, 255]) / 255  # #0598b0
dark_teal = np.array([5, 100, 110, 255]) / 255  # #0598b0
purple = np.array([138, 43, 226, 255]) / 255
grey = np.array([150, 150, 150, 255]) / 255
green = np.array([0, 143, 0, 255]) / 255
burnt_orange = np.array([133, 85, 67, 255]) / 255
light_orange = np.array([255, 207, 0, 255]) / 255

light_tan = np.array([255, 220, 170, 255]) / 255
tan = np.array([255, 190, 120, 255]) / 255

orchid = np.array([218, 112, 241, 255]) / 255
english_lavender = np.array([181, 135, 157, 255]) / 255

shadow_blue = np.array([119, 141, 169, 255]) / 255  # 778DA9
bedazzled_blue = np.array([65, 90, 119, 255]) / 255  # 415A77
oxford_blue = np.array([27, 38, 59, 255]) / 255  # 1B263B

cadet_blue = np.array([81., 163., 163., 255.]) / 255
eggplant = np.array([117., 72., 94., 255.]) / 255
persian_orange = np.array([203., 144., 77., 255.]) / 255
atomic_tangerine = np.array([247., 153., 110., 255.]) / 255
acrylide_yellow = np.array([233., 204., 116., 255.]) / 255

# Translucence in error bars
alpha_error = 0.2

SMALL = 8.5
MEDSMALL = 10
MEDIUM = 11
BIGGER = 14

PM = "\u00B1"


def load_nice_plot_rc(use_tex=True):
    plt.rc('font', size=BIGGER)  # Default
    plt.rc('axes', titlesize=MEDIUM)  # Axes titles
    plt.rc('axes', labelsize=MEDIUM)  # x and y labels
    plt.rc('xtick', labelsize=MEDSMALL)  # x tick labels
    plt.rc('ytick', labelsize=MEDSMALL)  # y tick labels
    plt.rc('legend', fontsize=MEDSMALL)  # Legend labels
    plt.rc('figure', titlesize=BIGGER)  # Figure title

    plt.rc('font', family='serif')
    plt.rc('font', serif='Palatino')
    plt.rc('figure', dpi=100)
    plt.rc('text', usetex=use_tex)


class MatplotlibVisProcess:
    def __init__(self, dt, init_fn, init_animate_fn, animate_fn):
        self.state_list = []
        self.state_names = []
        self.init_fn = init_fn  # create figures and maybe axes
        self.init_animate_fn = init_animate_fn  # animate init
        self.animate_fn = animate_fn  # animate loop

        self.dt = dt

    # register all of these before starting process
    def register_shared_state(self, state_name, state):
        self.state_list.append(state)
        self.state_names.append(state_name)

    def launch(self):
        self.vis_reading_remote, self.vis_work_remote = Pipe()
        # this reads from socket / sends to socket
        self.vis_proc = Process(target=MatplotlibVisProcess.run,
                                args=(self.vis_work_remote, self.vis_reading_remote, CloudPickleWrapper(self),
                                      self.state_names, *self.state_list))
        self.vis_proc.daemon = True  # if the main process crashes, we should not cause things to hang
        self.vis_proc.start()
        self.vis_work_remote.close()  # extra two-way socket

    @staticmethod
    def run(remote, parent_remote, self_wrapper, state_names, *state_list):
        mvp = self_wrapper.x
        logger.debug("Global visualization process beginning")
        parent_remote.close()

        dd = AttrDict()
        for i, name in enumerate(state_names):
            dd[name] = state_list[i]

        fig, other_data = mvp.init_fn(dd)

        anim = FuncAnimation(
            fig,
            lambda frame: mvp.animate_fn(frame, dd, other_data),
            init_func=lambda: mvp.init_animate_fn(dd, other_data),
            interval=mvp.dt,  # in ms
            blit=True,
            repeat=False
        )

        plt.show()


def get_axis_i(axes, i, rows, cols):
    if rows * cols == 1:
        return axes
    elif rows == 1 or cols == 1:
        return axes[i]
    else:
        return axes[i // cols, i % cols]


def equal_aspect_3d(ax, x, y, z, sgns=(1,1,1)):
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    zmin, zmax = z.min(), z.max()
    all_max = np.array([xmax, ymax, zmax])
    all_min = np.array([xmin, ymin, zmin])
    max_range = (all_max - all_min).max()
    center = 0.5 * (all_max + all_min)
    e_max = center + max_range / 2
    e_min = center - max_range / 2
    ax.set_xlim([e_min[0], e_max[0]][::sgns[0]])
    ax.set_ylim([e_min[1], e_max[1]][::sgns[1]])
    ax.set_zlim([e_min[2], e_max[2]][::sgns[2]])


def drawLineAnimations(queue, get_next_point_for_axis_fn, params):
    """
    when just plotting time-series data, use this and pipe the data in through the queue
    """
    get_next_point_for_axis_fn = get_next_point_for_axis_fn.__call__  # redundant tho
    matplotlib.use('TkAgg')
    # defaults here are for poses
    figsize = get_with_default(params, "figsize", (10, 6))
    tightlayout = get_with_default(params, "tightlayout", True)
    nrows = get_with_default(params, "nrows", 2)
    ncols = get_with_default(params, "ncols", 3)
    steps_to_keep = get_with_default(params, "steps_to_keep", 100)  # how many x steps to keep
    num_axes = nrows * ncols
    # how many outputs returned by get_next_point_for_axis_fn(queue.get()),
    #   which returns points for axis from queue output, (num_axes * lines_per_axis,)
    lines_per_axis = get_with_default(params, "lines_per_axis", 1)
    labels_per_axis = get_with_default(params, "labels_per_axis", None)
    ylim_padding = get_with_default(params, "ylim_tolerances", [1] * num_axes)
    # row major
    axis_titles = get_with_default(params, "axis_titles", [str(i) for i in range(num_axes)])
    assert len(axis_titles) == num_axes

    if labels_per_axis is not None:
        assert lines_per_axis == len(labels_per_axis), "must have same labels per axis"

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, tight_layout=tightlayout)

    xdata = []
    all_data_by_axis = [[] for _ in range(num_axes)]
    all_lines_by_axis = [[] for _ in range(num_axes)]
    for i in range(num_axes):
        ax = get_axis_i(axes, i, nrows, ncols)
        ax.set_title(axis_titles[i])
        for j in range(lines_per_axis):
            if labels_per_axis is not None and labels_per_axis[j] is not None:
                ln, = ax.plot([], [], label=labels_per_axis[j])
            else:
                ln, = ax.plot([], [])
            all_lines_by_axis[i].append(ln)
            all_data_by_axis[i].append([])

        if labels_per_axis is not None:
            ax.legend()

    def init():
        return [ln for axis_lines in all_lines_by_axis for ln in axis_lines]

    def update(time_frame):
        nonlocal xdata
        all_new = [queue.get()]
        xdata.append(0 if len(xdata) == 0 else xdata[-1] + 1)
        while not queue.empty():
            all_new.append(queue.get())
            xdata.append(xdata[-1] + 1)

        # process the queue data into data for the lines
        for i, new_data in enumerate(all_new):
            # (num_axes * lines_per_axis,)
            point_arr = get_next_point_for_axis_fn(new_data).reshape(num_axes, lines_per_axis)
            for k in range(num_axes):
                for l in range(lines_per_axis):
                    # add l'th line new point
                    all_data_by_axis[k][l].append(point_arr[k][l])

        # truncate & plot
        xdata = xdata[-steps_to_keep:]
        for k in range(num_axes):
            ax = get_axis_i(axes, k, nrows, ncols)
            all_y_pts = []
            for l in range(lines_per_axis):
                # truncate
                all_data_by_axis[k][l] = all_data_by_axis[k][l][-steps_to_keep:]
                # shared xdata
                all_lines_by_axis[k][l].set_data(xdata, all_data_by_axis[k][l])
                # limits x dimension
                all_y_pts.extend(all_data_by_axis[k][l])

            first = max(-1, xdata[-1] - steps_to_keep)
            ax.set_xlim(first, xdata[-1])
            ax.set_ylim(min(all_y_pts) - ylim_padding[k], max(all_y_pts) + ylim_padding[k])

        return [ln for axis_lines in all_lines_by_axis for ln in axis_lines]

    ani = FuncAnimation(fig, update, interval=10,
                        init_func=init, blit=False)

    plt.show()


# TODO
def drawLines(data: np.ndarray, params=AttrDict()):
    """
    when just plotting time-series data after-the-fact, use this

    data: arrays of shape (T, L, dim)  # L is the number of lines per axis
    keys: leaf_keys to plot in each axis, for each axis
    plot_idxs: list of lists for each key, default is all

    params:
        - figsize, nrows, ncols for subplots
        - steps_to_keep
    """
    matplotlib.use('TkAgg')

    # if plot_idxs is not None:
    #     assert len(plot_idxs) == len(keys)
    #     dc = dc.leaf_copy()
    #     for key, idxs in zip(keys, plot_idxs):
    #         arr = dc >> key
    #         assert 0 < len(idxs) <= arr.shape[-1]
    #         dc[key] = arr[:, idxs]
    #
    # data = np.asarray(concatenate(dc, keys), axis=-1)

    # defaults here are for poses
    figsize = get_with_default(params, "figsize", (10, 6))
    nrows = get_with_default(params, "nrows", 2)
    ncols = get_with_default(params, "ncols", 3)
    show = get_with_default(params, "show", True)
    steps_to_keep = get_with_default(params, "steps_to_keep", data.shape[0])  # how many x steps to keep (default is all)
    num_axes = nrows * ncols
    assert data.shape[-1] <= num_axes, "Concatenated data dimension is bigger than num_axes"
    if data.shape[-1] < num_axes:
        data = pad_dims(data, [-1], [num_axes], after=True, delta=False)

    # how many outputs returned by get_next_point_for_axis_fn(queue.get()),
    #   which returns points for axis from queue output, (num_axes * lines_per_axis,)
    # lines_per_axis = get_with_default(params, "lines_per_axis", 1)
    horizon, lines_per_axis, dim = data.shape
    labels_per_axis = get_with_default(params, "labels_per_axis", None)
    ylim_padding = get_with_default(params, "ylim_tolerances", [1] * num_axes)
    # row major
    axis_titles = get_with_default(params, "axis_titles", [str(i) for i in range(num_axes)])
    assert len(axis_titles) == num_axes

    if labels_per_axis is not None:
        assert lines_per_axis == len(labels_per_axis), "must have same labels per axis"

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    for i in range(num_axes):
        ax = get_axis_i(axes, i, nrows, ncols)
        ax.set_title(axis_titles[i])
        y_mins = []
        y_maxs = []
        all_lns = []
        for j in range(lines_per_axis):
            ydata = data[:, j, i]
            if labels_per_axis is not None and labels_per_axis[j] is not None:
                ln, = ax.plot(np.arange(horizon), ydata, label=labels_per_axis[j])
            else:
                ln, = ax.plot(np.arange(horizon), ydata)
            y_mins.append(np.min(ydata))
            y_maxs.append(np.max(ydata))
            all_lns.append(ln)

        if labels_per_axis is not None:
            ax.legend()

        first = max(-1, horizon - steps_to_keep)
        ax.set_xlim(first, horizon)
        ax.set_ylim(min(y_mins) - ylim_padding[i], max(y_maxs) + ylim_padding[i])

    if show:
        plt.show()


def make_legend(fig, labels, colors, marker='s', markersize=7, legend_type="below", ncol=None, above_by=0.1):
    """ fig: a matplotlib Figure object
        labels: a list of labels for the legend
        colors: the corresponding colors
        marker: shape of the legend markers (see matplotlib documentation)
        markersize: scaling of the legend markers
        legend_type: 'above' and 'below' currently defined, add more if you need
        ncol: number of columns in the legend
    """
    if ncol is None:
        ncol = len(labels)

    # Make fake line handles
    handles = [Line2D([0], [0], color='w', marker=marker, markerfacecolor=color, markersize=7) for color in colors]

    # Make a new legend type if you need further customization
    if legend_type == "below":
        legend = fig.legend(handles=handles,
                            labels=labels,
                            loc="lower center",
                            borderaxespad=1,
                            ncol=ncol,
                            bbox_to_anchor=(0.5, -0.1),
                            frameon=False
                            )
    elif legend_type == "above":
        legend = fig.legend(handles=handles,
                            labels=labels,
                            loc="upper center",
                            borderaxespad=1,
                            ncol=ncol,
                            bbox_to_anchor=(0.5, 1 + above_by),
                            frameon=False
                            )
    else:
        raise NotImplementedError

    return legend


def show(fig, legend, save=False, filename=None, dpi=50):
    fig.tight_layout()
    if not save:
        plt.show()
    else:
        fig.savefig(filename, bbox_extra_artists=(legend,), dpi=dpi, bbox_inches='tight')



if __name__ == '__main__':
    sample_data = np.arange(30)
    data = np.stack([sample_data * i for i in range(1, 13)], axis=-1).reshape((30, 2, 6))

    drawLines(data)
