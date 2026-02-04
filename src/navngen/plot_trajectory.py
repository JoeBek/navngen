import matplotlib.pyplot as plt
from load_trajecory import TrajectoryPoint, load_trajectory_from_txt
import numpy as np
from matplotlib.collections import LineCollection


def plot_trajectory(trajectories,
                    plane: str = "xz",
                    ax=None,
                    show: bool = True):
    """
    Plot bird's-eye view of trajectory translations.

    Accepts:
      - list[TrajectoryPoint] (preferred), or
      - list of (R, t) tuples where t is a 3-vector, or
      - numpy array of shape (N,3)

    - plane: which 2D plane to plot:
        'xz' (default) -> lateral (x) vs forward (z) top-down view,
        'xy' -> x vs y,
        'yz' -> y vs z.
    - ax: optional matplotlib Axes to draw on.
    - show: if True call plt.show().

    Returns the Axes used.
    """
    # Extract translations into (N,3) numpy array
    if isinstance(trajectories, np.ndarray):
        ts = np.asarray(trajectories, dtype=float)
        if ts.ndim == 1:
            ts = ts.reshape(1, -1)
    else:
        ts_list = []
        for item in trajectories:
            if isinstance(item, TrajectoryPoint):
                ts_list.append(np.asarray(item.position, dtype=float).reshape(3,))
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                # assume (R, t)
                t = item[1]
                ts_list.append(np.asarray(t, dtype=float).reshape(3,))
            else:
                # try to interpret item as a 3-vector
                ts_list.append(np.asarray(item, dtype=float).reshape(3,))
        ts = np.array(ts_list, dtype=float)

    if ts.ndim != 2 or ts.shape[1] < 2:
        raise ValueError("translations must be 3-vectors")

    if plane == "xz":
        xs, ys = ts[:, 0], ts[:, 2]
        xlabel, ylabel = "x", "z"
    elif plane == "xy":
        xs, ys = ts[:, 0], ts[:, 1]
        xlabel, ylabel = "x", "y"
    elif plane == "yz":
        xs, ys = ts[:, 1], ts[:, 2]
        xlabel, ylabel = "y", "z"
    else:
        raise ValueError("plane must be one of 'xz','xy','yz'")

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots()
        created_fig = True

    n = len(xs)
    if n < 1:
        return ax

    t = np.linspace(0.0, 1.0, n)
    colors = np.vstack((t, 1.0 - t, np.zeros_like(t))).T  # RGB: (r, g, 0)

    points = np.column_stack((xs, ys))
    if len(points) > 1:
        segments = np.stack([points[:-1], points[1:]], axis=1)
        seg_colors = colors[:-1]
        lc = LineCollection(segments, colors=seg_colors, linewidths=1.5)
        ax.add_collection(lc)
    else:
        ax.scatter(xs, ys, c=[colors[0]], s=20)

    ax.scatter(xs, ys, c=colors, s=20, edgecolors='face')
    ax.scatter(xs[0], ys[0], c=[colors[0]], s=60, label="start", edgecolors='k')
    ax.scatter(xs[-1], ys[-1], c=[colors[-1]], s=60, label="end", edgecolors='k')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True)
    ax.set_title("Visual Odometry on Airport Data")
    ax.legend()

    ax.relim()
    ax.autoscale_view()

    if show and created_fig:
        plt.show()

    return ax


if __name__ == '__main__':
    # Example usage: Create a dummy trajectory file and load it
    dummy_file_path = "/home/joe/nav_2/ORB_SLAM3/Examples/Monocular/CameraTrajectory.txt"
    print(f"Loading trajectory from {dummy_file_path}...")
    trajectory = load_trajectory_from_txt(dummy_file_path)

    plot_trajectory(trajectory, plane='xy')
    