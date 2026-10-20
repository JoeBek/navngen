"""
Run filter_trajectory.py on the airport trial (MSNV_*).

Builds a staging directory containing:
    images/       <- symlinks to downsampled_output_10fps_undistorted/frames/forward/*.jpg
    times.txt     <- one float timestamp per line (from rgb.txt first column)
    depth_masks/  <- symlinks to .../frames/depth_masks/*.npy  (depth/both mode)
    seg_masks/    <- symlinks to .../frames/seg_masks/*.npy    (seg/both mode)
"""

import argparse
import subprocess
import sys
import yaml
from pathlib import Path


DEFAULT_TRIAL_DIR = Path('/home/joe/data/vt/airport/MSNV_0000_0000_0_250804_1853_00156')
DEFAULT_CALIB     = Path('/home/joe/vt/research/glue/gluetest/configs/camera_forward.yaml')


def prepare_staging(trial_dir: Path, staging_dir: Path,
                    use_depth: bool, use_seg: bool) -> Path | None:
    frames_base = trial_dir / 'downsampled_output_10fps_undistorted' / 'frames'
    forward_dir = frames_base / 'forward'

    if not forward_dir.exists():
        print(f"Frames directory not found: {forward_dir}")
        return None

    img_files = sorted(forward_dir.glob('*.jpg'))
    if not img_files:
        print(f"No .jpg files found in {forward_dir}")
        return None

    img_dir = staging_dir / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)
    for src in img_files:
        dst = img_dir / src.name
        if not dst.is_symlink():
            dst.symlink_to(src.resolve())

    # rgb.txt is TUM format: "timestamp filename" — extract first column
    rgb_txt = frames_base / 'rgb.txt'
    if not rgb_txt.exists():
        print(f"rgb.txt not found at {rgb_txt}")
        return None
    times_dst = staging_dir / 'times.txt'
    with open(rgb_txt) as f:
        lines = [l for l in f if not l.startswith('#') and l.strip()]
    with open(times_dst, 'w') as f:
        for line in lines:
            parts = line.strip().split()
            if parts:
                f.write(parts[0] + '\n')

    if use_depth:
        dmask_src = frames_base / 'depth_masks'
        if not dmask_src.exists() or not any(dmask_src.iterdir()):
            print(f"depth_masks directory missing or empty: {dmask_src}")
            return None
        dmask_dir = staging_dir / 'depth_masks'
        dmask_dir.mkdir(exist_ok=True)
        for src in sorted(dmask_src.glob('*.npy')):
            dst = dmask_dir / src.name
            if not dst.is_symlink():
                dst.symlink_to(src.resolve())

    if use_seg:
        smask_src = frames_base / 'seg_masks'
        if not smask_src.exists() or not any(smask_src.iterdir()):
            print(f"seg_masks directory missing or empty: {smask_src}")
            return None
        smask_dir = staging_dir / 'seg_masks'
        smask_dir.mkdir(exist_ok=True)
        for src in sorted(smask_src.glob('*.npy')):
            dst = smask_dir / src.name
            if not dst.is_symlink():
                dst.symlink_to(src.resolve())

    return staging_dir


def main():
    project_root       = Path(__file__).resolve().parents[1]
    filter_script_path = project_root / 'scripts' / 'filter_trajectory.py'
    default_config     = project_root / 'scripts' / 'configs' / 'filter_config.yaml'
    default_staging    = project_root / 'assets' / 'staging' / 'airport'
    default_output     = project_root / 'assets' / 'outputs' / 'airport'

    if not filter_script_path.exists():
        print(f"Error: filter_trajectory.py not found at {filter_script_path}")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Run filtering for the airport trial.")
    parser.add_argument("--filter-mode", type=str, choices=['depth', 'segmentation', 'both'],
                        default='both',
                        help="Filtering mode.")
    parser.add_argument("--trial-dir", type=Path, default=DEFAULT_TRIAL_DIR,
                        help="Path to the airport trial directory.")
    parser.add_argument("--calib", type=Path, default=DEFAULT_CALIB,
                        help="Path to the surfnav camera YAML config.")
    parser.add_argument("--output_dir", type=Path, default=default_output,
                        help="Directory to save output trajectory file.")
    parser.add_argument("--staging_dir", type=Path, default=default_staging,
                        help="Directory for the staging dir.")
    parser.add_argument("--config_path", type=Path, default=default_config,
                        help="Path to filter_config.yaml for depth/seg thresholds.")
    parser.add_argument("--pickle_dir", type=Path, default=None,
                        help="If set, save processed frames as a .pkl.gz file here.")
    args = parser.parse_args()

    try:
        with open(args.config_path) as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: config file not found at {args.config_path}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.staging_dir.mkdir(parents=True, exist_ok=True)

    use_depth = args.filter_mode in ('depth', 'both')
    use_seg   = args.filter_mode in ('segmentation', 'both')

    staging_dir = prepare_staging(args.trial_dir, args.staging_dir, use_depth, use_seg)
    if staging_dir is None:
        sys.exit(1)

    trial_name  = args.trial_dir.name
    output_traj = args.output_dir / f"airport_{trial_name}_{args.filter_mode}_filtered_traj.txt"

    command = [
        "python", str(filter_script_path),
        "--input-path",    str(staging_dir),
        "--config_path",   str(args.calib),
        "--config_type",   "surfnav",
        "--image_dirname", "images",
        "--output_path",   str(output_traj),
    ]

    if args.pickle_dir:
        args.pickle_dir.mkdir(parents=True, exist_ok=True)
        pickle_path = args.pickle_dir / f"airport_{trial_name}_{args.filter_mode}_frames.pkl.gz"
        command.extend(["--pickle_path", str(pickle_path)])

    if args.filter_mode == 'depth':
        depth_cfg = config.get('depth', {})
        command.extend([
            "depth",
            "--mask-path", str(staging_dir / 'depth_masks'),
            "--tl", str(depth_cfg.get('tl', 0.0)),
            "--th", str(depth_cfg.get('th', 0.5)),
        ])
        if depth_cfg.get('normalize', False):
            command.append("--normalize")

    elif args.filter_mode == 'segmentation':
        seg_cfg = config.get('segmentation', {})
        command.extend([
            "segmentation",
            "--mask-path",  str(staging_dir / 'seg_masks'),
            "--filter-ids", str(seg_cfg.get('filter_ids', '0,1,2,3,4,5,7,8,9')),
        ])

    elif args.filter_mode == 'both':
        depth_cfg = config.get('depth', {})
        seg_cfg   = config.get('segmentation', {})
        command.extend([
            "both",
            "--depth-mask-path", str(staging_dir / 'depth_masks'),
            "--seg-mask-path",   str(staging_dir / 'seg_masks'),
            "--tl",              str(depth_cfg.get('tl', 0.0)),
            "--th",              str(depth_cfg.get('th', 0.5)),
            "--filter-ids",      str(seg_cfg.get('filter_ids', '0,1,2,3,4,5,7,8,9')),
        ])
        if depth_cfg.get('normalize', True):
            command.append("--normalize")

    print(f"Running filter ({args.filter_mode}) on {trial_name}...")
    try:
        subprocess.run(command, check=True, text=True)
        print(f"Done. Output: {output_traj}")
    except subprocess.CalledProcessError as e:
        print(f"Filter failed with return code {e.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
