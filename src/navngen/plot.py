import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from .frame import Frame
from typing import Sequence


def plot_trajectory(trajectory:Sequence[Frame],
                    plane: str = "xz",
                    ax=None,
                    show: bool = True):
    """
    Plot bird's-eye view of trajectory translations.

    Accepts:
    - list of Frame objects. each frame has lots of information, as defined in frame.py

    - plane: which 2D plane to plot:
        'xz' (default) -> lateral (x) vs forward (z) top-down view,
        'xy' -> x vs y,
        'yz' -> y vs z.
    - ax: optional matplotlib Axes to draw on.
    - show: if True call plt.show().

    Returns the Axes used.
    """
    # Extract translations into (N,3) numpy array
    ts_list = []
    for frame_item in trajectory: # Renamed loop variable for clarity
        if not isinstance(frame_item, Frame):
            raise TypeError("All items in trajectory must be Frame objects when using this function signature.")
        if frame_item.pose is None:
            raise ValueError("Frame object must have a pose set to be plotted.")
        t = frame_item.pose[1] # Extract translation from (R, t) tuple
        ts_list.append(np.asarray(t, dtype=float).reshape(3,))
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
    # Example usage: Create a dummy trajectory of Frame objects
    dummy_trajectory = []
    # Create some dummy Frame objects with poses
    for i in range(10):
        # Dummy rotation matrix (identity)
        R = np.eye(3)
        # Dummy translation vector
        t = np.array([i * 0.1, np.sin(i * 0.5) * 0.2, i * 0.2])
        dummy_frame = Frame(pose=(R, t))
        dummy_trajectory.append(dummy_frame)

    print("Plotting dummy trajectory of Frame objects...")
    plot_trajectory(dummy_trajectory, plane='xy')
    plt.title("Dummy Frame Trajectory (XY-plane)")
    plt.show()
    