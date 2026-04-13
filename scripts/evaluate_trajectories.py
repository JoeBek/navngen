#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
import tempfile
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Add project root so we can import navngen helpers
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.navngen.load_trajecory import load_ground_truth_euroc


def add_title_to_plot(png_path: Path, title: str):
    """Load a PNG saved by evo, add a suptitle, and overwrite it."""
    img = mpimg.imread(str(png_path))
    fig, ax = plt.subplots(figsize=(img.shape[1] / 100, img.shape[0] / 100 + 0.5), dpi=100)
    ax.imshow(img)
    ax.axis('off')
    fig.suptitle(title, fontsize=11, fontweight='bold', y=1.01)
    fig.savefig(str(png_path), bbox_inches='tight', dpi=100)
    plt.close(fig)

# EuRoC sequence name → subdirectory under euroc root
EUROC_CATEGORIES = {
    'MH': 'machine_hall',
    'V1': 'vicon_room1',
    'V2': 'vicon_room2',
}


def _euroc_gt_path(seq_name: str, euroc_root: Path) -> Path:
    prefix = seq_name[:2]
    category = EUROC_CATEGORIES.get(prefix)
    if category is None:
        raise ValueError(f"Unrecognised EuRoC sequence prefix '{prefix}' in '{seq_name}'")
    return euroc_root / category / seq_name / 'mav0'


def evaluate_kitti(args):
    input_dir = Path(args.input_dir)
    gt_dir = Path(args.gt_dir)
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(args.num_seq), desc="Evaluating trajectories"):
        seq = f"{i:02d}"
        gt_file = gt_dir / f"{seq}.txt"

        if not gt_file.exists():
            tqdm.write(f"Skipping seq {seq}, GT file {gt_file} not found")
            continue

        est_file = None
        for suffix in ["_traj.kitti", ".txt", ".kitti"]:
            test_file = input_dir / f"{seq}{suffix}"
            if test_file.exists():
                est_file = test_file
                break

        if not est_file:
            tqdm.write(f"Skipping seq {seq}, estimated file not found in {input_dir}")
            continue

        output_plot = plot_dir / f"{seq}_ape_xz"
        cmd = [
            "evo_ape", "kitti",
            str(gt_file), str(est_file),
            "-a", "-s",
            "--plot_mode", "xz",
            "--save_plot", str(output_plot),
            "--silent",
        ]
        if args.plot:
            cmd.append("-p")

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            tqdm.write(f"Error evaluating sequence {seq}: {e}")


def evaluate_euroc(args):
    input_dir = Path(args.input_dir)
    euroc_root = Path(args.euroc_data_path)
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Collect estimated trajectory files: *_traj.txt
    est_files = sorted(input_dir.glob("*_traj.txt"))
    if not est_files:
        print(f"No *_traj.txt files found in {input_dir}")
        sys.exit(1)

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"

    results = []  # list of (seq_name, ate_rmse, trel, rrel)

    for est_file in tqdm(est_files, desc="Evaluating EuRoC sequences"):
        # Filename is e.g. MH_01_easy_both_filtered_traj.txt or MH_01_easy_traj.txt
        # EuRoC sequence names are always 3 underscore-separated parts (e.g. MH_01_easy)
        seq_name = '_'.join(est_file.stem.split('_')[:3])

        try:
            mav0_path = _euroc_gt_path(seq_name, euroc_root)
        except ValueError as e:
            tqdm.write(f"[SKIP] {seq_name}: {e}")
            continue

        if not mav0_path.exists():
            tqdm.write(f"[SKIP] {seq_name}: mav0 not found at {mav0_path}")
            continue

        gt_tum = load_ground_truth_euroc(mav0_path)

        # Write GT to a temporary TUM file for evo
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.txt', delete=False, prefix=f"gt_{seq_name}_"
        ) as tmp:
            np.savetxt(tmp, gt_tum, fmt='%.9f')
            gt_tmp_path = tmp.name

        ate_rmse = trel_rmse = rrel_rmse = float('nan')
        try:
            # --- ATE ---
            plot_stem = plot_dir / f"{seq_name}_ape"
            ape_out = subprocess.run(
                ["evo_ape", "tum", gt_tmp_path, str(est_file),
                 "-a", "-s", "--plot_mode", "xz",
                 "--save_plot", str(plot_stem)],
                check=True, capture_output=True, text=True, env=env,
            )
            for line in ape_out.stdout.splitlines():
                if "rmse" in line:
                    ate_rmse = float(line.split()[-1])
                    break
            # Add method + sequence title to each saved PNG
            title = f"{args.method} — {seq_name}"
            for suffix in ("_map.png", "_raw.png"):
                p = Path(str(plot_stem) + suffix)
                if p.exists():
                    add_title_to_plot(p, title)

            # --- trel / rrel: averaged over sub-sequence lengths [1,2,5,10,20m] ---
            # This matches the multi-delta convention common in VO literature.
            deltas = [1, 2, 5, 10, 20]
            trel_vals = []
            rrel_vals = []
            for delta in deltas:
                try:
                    rpe_t = subprocess.run(
                        ["evo_rpe", "tum", gt_tmp_path, str(est_file),
                         "-a", "-s", "--pose_relation", "trans_part",
                         "--delta", str(delta), "--delta_unit", "m"],
                        check=True, capture_output=True, text=True, env=env,
                    )
                    rpe_r = subprocess.run(
                        ["evo_rpe", "tum", gt_tmp_path, str(est_file),
                         "-a", "-s", "--pose_relation", "angle_deg",
                         "--delta", str(delta), "--delta_unit", "m"],
                        check=True, capture_output=True, text=True, env=env,
                    )
                except subprocess.CalledProcessError:
                    # Trajectory too short for this delta — skip it
                    continue
                for line in rpe_t.stdout.splitlines():
                    if "rmse" in line:
                        trel_vals.append(float(line.split()[-1]) / delta * 100)
                        break
                for line in rpe_r.stdout.splitlines():
                    if "rmse" in line:
                        rrel_vals.append(float(line.split()[-1]) / delta)
                        break
            trel_rmse = float(np.mean(trel_vals)) if trel_vals else float('nan')
            rrel_rmse = float(np.mean(rrel_vals)) if rrel_vals else float('nan')

            results.append((seq_name, ate_rmse, trel_rmse, rrel_rmse))
            tqdm.write(f"[OK] {seq_name}  ATE={ate_rmse:.4f}m  trel={trel_rmse:.4f}%  rrel={rrel_rmse:.4f}deg/m")

        except subprocess.CalledProcessError as e:
            tqdm.write(f"[ERROR] {seq_name}: return code {e.returncode}")
            tqdm.write(e.stderr[-500:] if e.stderr else "")
        finally:
            os.unlink(gt_tmp_path)

    # Write summary table
    if results:
        summary_path = Path(args.input_dir) / "results.txt"
        with open(summary_path, 'w') as f:
            f.write(f"{'sequence':<25} {'ATE (m)':>10} {'trel (%)':>10} {'rrel (deg/m)':>13}\n")
            f.write("-" * 62 + "\n")
            for seq, ate, trel, rrel in results:
                f.write(f"{seq:<25} {ate:>10.4f} {trel:>10.4f} {rrel:>13.4f}\n")
            ates  = [r[1] for r in results]
            trels = [r[2] for r in results]
            rrels = [r[3] for r in results]
            f.write("-" * 62 + "\n")
            f.write(f"{'mean':<25} {np.mean(ates):>10.4f} {np.mean(trels):>10.4f} {np.mean(rrels):>13.4f}\n")
        print(f"\nResults saved to {summary_path}")


CAMPUS_GT_ROOT = Path('/home/joe/data/vt/airport/campus/trials_undistorted')
CAMPUS_TRAJ_ROOT = Path('/home/joe/vt/research/glue/outputs/trajectories/vt/campus')
CAMPUS_TRIALS = [10, 11, 12]
AIRPORT_DIR = Path('/home/joe/vt/research/glue/outputs/trajectories/vt/airport')


def evaluate_campus(args):
    """ATE RMSE for LG-VO and ORB-SLAM on campus trials, aligned/scaled to GT."""
    output_path = Path(args.output)
    env = os.environ.copy()
    env['MPLBACKEND'] = 'Agg'

    results = []  # (trial, our_ate, orb_ate)

    for trial in tqdm(CAMPUS_TRIALS, desc='Campus'):
        trial_dir  = CAMPUS_TRAJ_ROOT / str(trial)
        our_traj   = trial_dir / f'campus_trial_{trial}_both_filtered_traj.txt'
        orb_traj   = trial_dir / f'campus_{trial}_orbslam_mono.tum'
        gt_file    = CAMPUS_GT_ROOT / f'trial_{trial:02d}' / 'ground_truth_trajectory_vision.txt'

        missing = [n for n, p in [('ours', our_traj), ('orb', orb_traj), ('gt', gt_file)]
                   if not p.exists()]
        if missing:
            tqdm.write(f'[SKIP] trial {trial}: missing {missing}')
            continue

        # Our pipeline timestamps are in seconds; GT/ORB-SLAM are in nanosecond-scale.
        # Rescale ours by 1e9 so evo can match timestamps.
        our_data = np.loadtxt(our_traj)
        if our_data.ndim == 1:
            our_data = our_data[np.newaxis, :]
        our_data[:, 0] *= 1e9

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
            np.savetxt(tmp, our_data, fmt='%.9f')
            our_scaled = Path(tmp.name)

        our_ate = orb_ate = float('nan')
        try:
            for label, traj_path in [(args.method, our_scaled), ('ORB-SLAM', orb_traj)]:
                r = subprocess.run(
                    ['evo_ape', 'tum', str(gt_file), str(traj_path), '-a', '-s'],
                    check=True, capture_output=True, text=True, env=env,
                )
                for line in r.stdout.splitlines():
                    if 'rmse' in line:
                        val = float(line.split()[-1])
                        if label == args.method:
                            our_ate = val
                        else:
                            orb_ate = val
                        break

            results.append((trial, our_ate, orb_ate))
            tqdm.write(f'[OK] trial {trial}  {args.method}={our_ate:.4f}m  ORB-SLAM={orb_ate:.4f}m')
        except subprocess.CalledProcessError as e:
            tqdm.write(f'[ERROR] trial {trial}: {e.returncode}\n{(e.stderr or "")[-400:]}')
        finally:
            our_scaled.unlink(missing_ok=True)

    if results:
        with open(output_path, 'w') as f:
            f.write(f"{'trial':<10} {args.method+' ATE (m)':>18} {'ORB-SLAM ATE (m)':>18}\n")
            f.write('-' * 50 + '\n')
            for trial, our, orb in results:
                f.write(f"{trial:<10} {our:>18.4f} {orb:>18.4f}\n")
            ours = [r[1] for r in results]
            orbs = [r[2] for r in results]
            f.write('-' * 50 + '\n')
            f.write(f"{'mean':<10} {np.mean(ours):>18.4f} {np.mean(orbs):>18.4f}\n")
        print(f'\nResults saved to {output_path}')


def main():
    parser = argparse.ArgumentParser(description="Evaluate trajectories with evo_ape")
    parser.add_argument("input_dir", nargs='?', type=str,
                        help="Directory containing estimated trajectories (kitti/euroc modes)")
    parser.add_argument("--dataset", type=str, choices=["kitti", "euroc", "campus"], default="kitti",
                        help="Dataset format (default: kitti)")
    # KITTI options
    parser.add_argument("--gt_dir", type=str, default="assets/outputs/kitti/gt",
                        help="[KITTI] Directory containing GT trajectories")
    parser.add_argument("--num_seq", type=int, default=10,
                        help="[KITTI] Number of sequences to evaluate (default 10 for 00-09)")
    # EuRoC options
    parser.add_argument("--euroc_data_path", type=str, default="/home/joe/data/euroc",
                        help="[EuRoC] Path to the root EuRoC data directory")
    # Campus options
    parser.add_argument("--output", type=str, default="campus_results.txt",
                        help="[Campus] Output text file for ATE results")
    # Common options
    parser.add_argument("--plot_dir", type=str, default="assets/plots",
                        help="Directory to save plots")
    parser.add_argument("--plot", action="store_true", help="Show plot window")
    parser.add_argument("--method", type=str, default="LG-VO",
                        help="Method name shown in plot titles (default: LG-VO)")
    args = parser.parse_args()

    if args.dataset == "kitti":
        evaluate_kitti(args)
    elif args.dataset == "euroc":
        evaluate_euroc(args)
    else:
        evaluate_campus(args)


if __name__ == "__main__":
    main()
