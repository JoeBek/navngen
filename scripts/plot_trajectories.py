"""
Plot LG-VO vs ORB-SLAM vs GT for EuRoC, KITTI, and Campus using evo_traj.
All plots are in the xz plane, Sim(3)-aligned to GT.

Output: ~/vt/research/glue/outputs/plots/{euroc,kitti,campus}/

Usage:
  python plot_trajectories.py
  python plot_trajectories.py --method "LG-VO+depth+seg"
"""

import os
import sys
import shutil
import subprocess
import argparse
import tempfile
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
from src.navngen.load_trajecory import load_ground_truth_euroc

TRAJ_ROOT    = Path('/home/joe/vt/research/glue/outputs/trajectories')
KITTI_GT_DIR = project_root / 'assets/outputs/kitti/gt'
PLOT_ROOT    = Path('/home/joe/vt/research/glue/outputs/plots')
EUROC_DATA   = Path('/home/joe/data/euroc')

EUROC_CATS = {'MH': 'machine_hall', 'V1': 'vicon_room1', 'V2': 'vicon_room2'}
EUROC_SEQS = [
    'MH_01_easy', 'MH_02_easy', 'MH_03_medium', 'MH_04_difficult', 'MH_05_difficult',
    'V1_01_easy', 'V1_02_medium', 'V1_03_difficult',
    'V2_01_easy', 'V2_02_medium', 'V2_03_difficult',
]

ENV = {**os.environ, 'MPLBACKEND': 'Agg'}


def tum_to_kitti(tum_path: Path, kitti_path: Path):
    """Convert TUM trajectory (8 cols) to KITTI format (12 cols, 3x4 matrix)."""
    data = np.loadtxt(tum_path)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    with open(kitti_path, 'w') as f:
        for row in data:
            t = row[1:4]
            q = row[4:8]  # qx qy qz qw
            R = Rotation.from_quat(q).as_matrix()
            pose = np.hstack([R, t.reshape(3, 1)]).flatten()
            f.write(' '.join(f'{v:.6e}' for v in pose) + '\n')


def run_evo_traj(fmt, named_files: dict, ref_name: str | None, plot_path: Path,
                 align=True, scale=True, align_origin=False):
    """
    named_files: {label: Path} — written into a temp dir so evo uses label as filename.
    ref_name: key in named_files to use as --ref (or None).
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        staged = {}
        for label, src in named_files.items():
            dst = tmp / f'{label}.txt'
            shutil.copy(src, dst)
            staged[label] = dst

        traj_files = [str(p) for label, p in staged.items() if label != ref_name]
        cmd = ['evo_traj', fmt] + traj_files
        if ref_name and ref_name in staged:
            cmd += ['--ref', str(staged[ref_name])]
        if align_origin:
            cmd.append('--align_origin')
        elif align:
            cmd.append('-a')
            if scale:
                cmd.append('-s')
        cmd += ['--plot_mode', 'xz', '--save_plot', str(plot_path)]
        subprocess.run(cmd, check=True, env=ENV, capture_output=True, text=True)


# ── EuRoC ─────────────────────────────────────────────────────────────────────

def plot_euroc(method_name: str):
    out_dir = PLOT_ROOT / 'euroc'
    out_dir.mkdir(parents=True, exist_ok=True)

    for seq in tqdm(EUROC_SEQS, desc='EuRoC'):
        our_traj    = TRAJ_ROOT / f'{seq}_both_filtered_traj.txt'
        orbslam_traj = TRAJ_ROOT / 'euroc' / f'CameraTrajectory_{seq}.txt'
        mav0 = EUROC_DATA / EUROC_CATS[seq[:2]] / seq / 'mav0'

        missing = [n for n, p in [('ours', our_traj), ('orb', orbslam_traj), ('mav0', mav0)]
                   if not p.exists()]
        if missing:
            tqdm.write(f'[SKIP] {seq}: missing {missing}')
            continue

        gt_tum = load_ground_truth_euroc(mav0)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
            np.savetxt(tmp, gt_tum, fmt='%.9f')
            gt_tmp = Path(tmp.name)

        try:
            run_evo_traj(
                'tum',
                named_files={'GT': gt_tmp, method_name: our_traj, 'ORB-SLAM': orbslam_traj},
                ref_name='GT',
                plot_path=out_dir / seq,
            )
            tqdm.write(f'[OK] {seq}')
        except subprocess.CalledProcessError as e:
            tqdm.write(f'[ERROR] {seq}: {e.returncode}\n{e.stderr[-300:]}')
        finally:
            gt_tmp.unlink(missing_ok=True)


# ── KITTI ─────────────────────────────────────────────────────────────────────

def plot_kitti(method_name: str):
    out_dir = PLOT_ROOT / 'kitti'
    out_dir.mkdir(parents=True, exist_ok=True)
    orbslam_dir = TRAJ_ROOT / 'kitti' / 'orbslam'

    for i in tqdm(range(11), desc='KITTI'):
        seq = f'{i:02d}'
        our_traj     = TRAJ_ROOT / f'kitti_{seq}_both_filtered_traj.txt'
        orbslam_tum  = orbslam_dir / f'{seq}_orbslam_mono.tum'
        gt_file      = KITTI_GT_DIR / f'{seq}.txt'

        missing = [n for n, p in [('ours', our_traj), ('orb', orbslam_tum), ('gt', gt_file)]
                   if not p.exists()]
        if missing:
            tqdm.write(f'[SKIP] KITTI {seq}: missing {missing}')
            continue

        # Convert ORB-SLAM TUM → KITTI format
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            orbslam_kitti = Path(tmp.name)
        tum_to_kitti(orbslam_tum, orbslam_kitti)

        try:
            # KITTI trajectories can have different pose counts (ORB-SLAM tracking losses),
            # so Umeyama alignment fails. Use origin alignment instead.
            run_evo_traj(
                'kitti',
                named_files={'GT': gt_file, method_name: our_traj, 'ORB-SLAM': orbslam_kitti},
                ref_name='GT',
                plot_path=out_dir / f'kitti_{seq}',
                align=False,
                scale=False,
                align_origin=True,
            )
            tqdm.write(f'[OK] KITTI {seq}')
        except subprocess.CalledProcessError as e:
            tqdm.write(f'[ERROR] KITTI {seq}: {e.returncode}\n{e.stderr[-300:]}')
        finally:
            orbslam_kitti.unlink(missing_ok=True)


# ── Campus ────────────────────────────────────────────────────────────────────

CAMPUS_GT_ROOT = Path('/home/joe/data/vt/airport/campus/trials_undistorted')


def rescale_timestamps(src: Path, dst: Path, factor: float):
    """Write a copy of a TUM file with timestamps multiplied by factor."""
    data = np.loadtxt(src)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    data[:, 0] *= factor
    np.savetxt(dst, data, fmt='%.9f')


def plot_campus(method_name: str):
    out_dir = PLOT_ROOT / 'campus'
    out_dir.mkdir(parents=True, exist_ok=True)
    campus_root = TRAJ_ROOT / 'vt' / 'campus'

    for trial in tqdm([10, 11, 12], desc='Campus'):
        trial_dir    = campus_root / str(trial)
        our_traj     = trial_dir / f'campus_trial_{trial}_both_filtered_traj.txt'
        orbslam_traj = trial_dir / f'campus_{trial}_orbslam_mono.tum'
        gt_file      = CAMPUS_GT_ROOT / f'trial_{trial:02d}' / 'ground_truth_trajectory_vision.txt'

        missing = [n for n, p in [('ours', our_traj), ('orb', orbslam_traj), ('gt', gt_file)]
                   if not p.exists()]
        if missing:
            tqdm.write(f'[SKIP] Campus {trial}: missing {missing}')
            continue

        # Our pipeline outputs timestamps in seconds (divided by 1e9), while GT/ORB-SLAM
        # use nanosecond-scale values. Rescale ours to match before passing to evo.
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            our_rescaled = Path(tmp.name)
        rescale_timestamps(our_traj, our_rescaled, factor=1e9)

        try:
            run_evo_traj(
                'tum',
                named_files={'GT': gt_file, method_name: our_rescaled, 'ORB-SLAM': orbslam_traj},
                ref_name='GT',
                plot_path=out_dir / f'campus_{trial}',
            )
            tqdm.write(f'[OK] Campus {trial}')
        except subprocess.CalledProcessError as e:
            tqdm.write(f'[ERROR] Campus {trial}: {e.returncode}\n{e.stderr[-300:]}')
        finally:
            our_rescaled.unlink(missing_ok=True)


# ── Airport ───────────────────────────────────────────────────────────────────

AIRPORT_DIR = TRAJ_ROOT / 'vt' / 'airport'


def plot_airport(method_name: str):
    out_dir = PLOT_ROOT / 'airport'
    out_dir.mkdir(parents=True, exist_ok=True)

    our_traj     = AIRPORT_DIR / 'ap_seg.txt'
    orbslam_traj = AIRPORT_DIR / 'ap_orbslam_mono.tum'
    gt_file      = AIRPORT_DIR / 'groundtruth.txt'

    missing = [n for n, p in [('ours', our_traj), ('orb', orbslam_traj), ('gt', gt_file)]
               if not p.exists()]
    if missing:
        print(f'[SKIP] Airport: missing {missing}')
        return

    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
        our_rescaled = Path(tmp.name)
    rescale_timestamps(our_traj, our_rescaled, factor=1e9)

    try:
        run_evo_traj(
            'tum',
            named_files={'GT': gt_file, method_name: our_rescaled, 'ORB-SLAM': orbslam_traj},
            ref_name='GT',
            plot_path=out_dir / 'airport',
        )
        print('[OK] Airport')
    except subprocess.CalledProcessError as e:
        print(f'[ERROR] Airport: {e.returncode}\n{e.stderr[-300:]}')
    finally:
        our_rescaled.unlink(missing_ok=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Plot LG-VO vs ORB-SLAM vs GT trajectories.')
    parser.add_argument('--method', type=str, default='LG-VO',
                        help='Label for our method in plots (default: LG-VO)')
    args = parser.parse_args()

    plot_euroc(args.method)
    plot_kitti(args.method)
    plot_campus(args.method)
    plot_airport(args.method)
    print(f'\nAll plots saved to {PLOT_ROOT}')


if __name__ == '__main__':
    main()
