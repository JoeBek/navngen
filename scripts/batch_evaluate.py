"""
Batch trajectory evaluation using evo.

For each estimated trajectory in --traj-dir, finds a matching GT file in --gt-dir
and computes ATE (APE translation RMSE), RTE_t (RPE translation RMSE), and
RTE_r (RPE rotation RMSE in degrees). Results are written to a text file.

Both GT and estimated trajectory formats are auto-detected from the first line
(12 values = KITTI, 8 = TUM).

GT matching: looks for a GT file whose stem appears as a substring of the
estimated filename stem (e.g. gt/00.txt matches kitti_00_depth_filtered_traj.txt).

Alignment: Umeyama (no scale correction) applied by default before APE/RPE.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


def detect_format(path: Path) -> str:
    """Return 'kitti' (12 values/line) or 'tum' (8 values/line)."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                return 'kitti' if len(line.split()) == 12 else 'tum'
    raise ValueError(f"Cannot detect format of {path}")


def load_trajectory(path: Path, fmt: str):
    from evo.tools import file_interface
    if fmt == 'kitti':
        return file_interface.read_kitti_poses_file(str(path))
    else:
        return file_interface.read_tum_trajectory_file(str(path))


def to_timestamped(traj, n: int):
    """Convert any evo trajectory to PoseTrajectory3D with index timestamps, truncated to n."""
    from evo.core.trajectory import PoseTrajectory3D
    poses = traj.poses_se3[:n]
    return PoseTrajectory3D(poses_se3=poses, timestamps=np.arange(n, dtype=float))


STAT_KEYS = ['rmse', 'mean', 'median', 'std', 'min', 'max']


def get_stats(metric) -> dict:
    from evo.core.metrics import StatisticsType
    return {k: metric.get_statistic(getattr(StatisticsType, k)) for k in STAT_KEYS}


def evaluate(gt_path: Path, est_path: Path, align: bool, scale: bool = False, plot_path: Path | None = None):
    from evo.core.metrics import APE, RPE, PoseRelation

    gt_fmt  = detect_format(gt_path)
    est_fmt = detect_format(est_path)
    gt_raw  = load_trajectory(gt_path,  gt_fmt)
    est_raw = load_trajectory(est_path, est_fmt)

    n = min(len(gt_raw.poses_se3), len(est_raw.poses_se3))
    if n < 2:
        raise ValueError(f"Too few poses ({n}) after truncation")

    gt_t  = to_timestamped(gt_raw,  n)
    est_t = to_timestamped(est_raw, n)

    if align:
        est_t.align(gt_t, correct_scale=scale)

    ape = APE(PoseRelation.translation_part)
    ape.process_data((gt_t, est_t))
    ate_stats = get_stats(ape)

    rpe_t = RPE(PoseRelation.translation_part)
    rpe_t.process_data((gt_t, est_t))
    rte_t_stats = get_stats(rpe_t)

    rpe_r = RPE(PoseRelation.rotation_angle_deg)
    rpe_r.process_data((gt_t, est_t))
    rte_r_stats = get_stats(rpe_r)

    if plot_path is not None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from evo.tools import plot as evo_plot

        fig = plt.figure()
        evo_plot.trajectories(fig, {'GT': gt_t, 'Est': est_t},
                              plot_mode=evo_plot.PlotMode.xz,
                              title=f"{est_path.stem}\n"
                                    f"ATE rmse={ate_stats['rmse']:.4f}  "
                                    f"RTE_t rmse={rte_t_stats['rmse']:.4f}  "
                                    f"RTE_r rmse={rte_r_stats['rmse']:.4f}")
        fig.savefig(plot_path, bbox_inches='tight')
        plt.close(fig)

    return ate_stats, rte_t_stats, rte_r_stats


def find_gt_file(est_stem: str, gt_dir: Path) -> Path | None:
    """Return the GT file whose stem is a substring of est_stem, or None."""
    gt_files = sorted(gt_dir.glob('*.txt'))
    # Prefer the longest matching stem to avoid false matches (e.g. '0' vs '00')
    matches = [f for f in gt_files if f.stem in est_stem]
    if not matches:
        return None
    return max(matches, key=lambda f: len(f.stem))


def main():
    project_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description='Batch evaluate trajectories with evo APE/RPE.')
    parser.add_argument('--traj-dir', '-t', type=Path, required=True,
                        help='Directory containing estimated TUM trajectory files.')
    parser.add_argument('--gt-dir', '-g', type=Path,
                        default=project_root / 'assets' / 'outputs' / 'kitti' / 'gt',
                        help='Directory containing ground truth trajectory files.')
    parser.add_argument('--output', '-o', type=Path,
                        default=None,
                        help='Output text file. Defaults to <traj-dir>/eval_results.txt.')
    parser.add_argument('--no-align', action='store_true',
                        help='Disable Umeyama alignment before evaluation.')
    parser.add_argument('--scale', action='store_true',
                        help='Also correct scale during Umeyama alignment.')
    parser.add_argument('--plot', action='store_true',
                        help='Save a trajectory plot (GT vs Est) for each evaluated file.')
    parser.add_argument('--plot-dir', type=Path,
                        default=project_root / 'assets' / 'plots' / 'dump',
                        help='Directory to save plots (default: assets/plots/dump).')
    args = parser.parse_args()

    if not args.traj_dir.is_dir():
        print(f"Error: traj-dir not found: {args.traj_dir}")
        sys.exit(1)
    if not args.gt_dir.is_dir():
        print(f"Error: gt-dir not found: {args.gt_dir}")
        sys.exit(1)

    output_path = args.output or (args.traj_dir / 'eval_results.txt')
    align = not args.no_align

    if args.plot:
        args.plot_dir.mkdir(parents=True, exist_ok=True)

    est_files = sorted(args.traj_dir.glob('*.txt'))
    if not est_files:
        print(f"No .txt files found in {args.traj_dir}")
        sys.exit(1)

    # Build header: filename + one column per stat per metric
    stat_labels = [f"{m}_{s}" for m in ('ATE', 'RTE_t', 'RTE_r') for s in STAT_KEYS]
    col_w = 10
    header = f"{'filename':<60}  " + "  ".join(f"{l:>{col_w}}" for l in stat_labels)
    rows = []

    for est_path in tqdm(est_files, desc="Evaluating"):
        gt_path = find_gt_file(est_path.stem, args.gt_dir)
        if gt_path is None:
            tqdm.write(f"  [SKIP] no GT match for {est_path.name}")
            continue

        try:
            plot_path = (args.plot_dir / f"{est_path.stem}.png") if args.plot else None
            ate_s, rte_t_s, rte_r_s = evaluate(gt_path, est_path, align, scale=args.scale, plot_path=plot_path)
            vals = [ate_s[s] for s in STAT_KEYS] + [rte_t_s[s] for s in STAT_KEYS] + [rte_r_s[s] for s in STAT_KEYS]
            row = f"{est_path.name:<60}  " + "  ".join(f"{v:>{col_w}.4f}" for v in vals)
            rows.append(row)
            tqdm.write(f"  {est_path.name}  ATE(rmse={ate_s['rmse']:.4f} mean={ate_s['mean']:.4f} med={ate_s['median']:.4f} min={ate_s['min']:.4f} max={ate_s['max']:.4f})")
        except Exception as e:
            tqdm.write(f"  [ERROR] {est_path.name}: {e}")
            rows.append(f"{est_path.name:<60}  {'ERROR':>{col_w}}")

    with open(output_path, 'w') as f:
        f.write(header + '\n')
        f.write('-' * len(header) + '\n')
        f.write('\n'.join(rows) + '\n')

    print(f"\nResults written to {output_path}")


if __name__ == '__main__':
    main()
