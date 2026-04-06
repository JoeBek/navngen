"""
Wrapper script to run filter_trajectory.py on campus trial directories,
mirroring the structure of run_nuscenes_filter.py.

For each trial this script:
  1. Builds a staging directory containing:
       images/          <- symlinks to trial frames/forward/*.jpg
       times.txt        <- per-frame timestamps (one float per line)
       depth_masks/     <- symlinks to frames/depth_masks/*.npy  (depth/both mode)
       seg_masks/       <- symlinks to frames/seg_masks/*.npy    (seg/both mode)
  2. Calls filter_trajectory.py via subprocess with --config_type surfnav.
"""

import argparse
import subprocess
import sys
import yaml
from pathlib import Path
from tqdm import tqdm


DEFAULT_TRIALS_DIR  = Path('/home/joe/data/vt/airport/campus/trials')
DEFAULT_CALIB       = Path('/home/joe/vt/research/glue/gluetest/configs/camera_forward.yaml')


def prepare_staging(trial_dir: Path, staging_root: Path,
                    use_depth: bool, use_seg: bool) -> Path | None:
    """
    Creates a staging directory for one campus trial.
    Returns the staging Path, or None if required masks are missing.
    """
    name = trial_dir.name
    staging_dir = staging_root / name
    img_dir = staging_dir / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)

    frames_dir = trial_dir / 'frames' / 'forward'
    if not frames_dir.exists():
        tqdm.write(f"[{name}] No frames/forward directory, skipping.")
        return None

    img_files = sorted(frames_dir.glob('*.jpg'))
    if not img_files:
        tqdm.write(f"[{name}] No .jpg files found, skipping.")
        return None

    # --- image symlinks ---
    for src in img_files:
        dst = img_dir / src.name
        if not dst.is_symlink():
            dst.symlink_to(src.resolve())

    # --- times.txt: extract second column from forward_times.txt ---
    times_src = trial_dir / 'frames' / 'forward_times.txt'
    times_dst = staging_dir / 'times.txt'
    if not times_src.exists():
        tqdm.write(f"[{name}] No forward_times.txt found, skipping.")
        return None
    with open(times_src) as f:
        lines = f.readlines()
    with open(times_dst, 'w') as f:
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                f.write(parts[1] + '\n')

    # --- depth mask symlinks ---
    if use_depth:
        dmask_src_dir = trial_dir / 'frames' / 'depth_masks'
        if not dmask_src_dir.exists() or not any(dmask_src_dir.iterdir()):
            tqdm.write(f"[{name}] No depth_masks directory found, skipping.")
            return None
        dmask_dir = staging_dir / 'depth_masks'
        dmask_dir.mkdir(exist_ok=True)
        for src in dmask_src_dir.glob('*.npy'):
            dst = dmask_dir / src.name
            if not dst.is_symlink():
                dst.symlink_to(src.resolve())

    # --- seg mask symlinks ---
    if use_seg:
        smask_src_dir = trial_dir / 'frames' / 'seg_masks'
        if not smask_src_dir.exists() or not any(smask_src_dir.iterdir()):
            tqdm.write(f"[{name}] No seg_masks directory found, skipping.")
            return None
        smask_dir = staging_dir / 'seg_masks'
        smask_dir.mkdir(exist_ok=True)
        for src in smask_src_dir.glob('*.npy'):
            dst = smask_dir / src.name
            if not dst.is_symlink():
                dst.symlink_to(src.resolve())

    return staging_dir


def main():
    project_root       = Path(__file__).resolve().parents[1]
    filter_script_path = project_root / 'scripts' / 'filter_trajectory.py'
    default_save_path  = project_root / 'assets' / 'outputs' / 'campus'
    default_config     = project_root / 'scripts' / 'configs' / 'filter_config.yaml'
    default_staging    = project_root / 'assets' / 'staging' / 'campus'

    if not filter_script_path.exists():
        print(f"Error: filter_trajectory.py not found at {filter_script_path}")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Run filtering for campus trial directories.")
    parser.add_argument("--filter-mode", type=str, choices=['depth', 'segmentation', 'both'],
                        default='depth',
                        help="Filtering mode. 'both' applies depth then segmentation.")
    parser.add_argument("--trials-dir", type=Path, default=DEFAULT_TRIALS_DIR,
                        help="Path to the campus/trials directory.")
    parser.add_argument("--trials", type=str, nargs='+', default=None,
                        help="Specific trial names to process (e.g. trial_00 trial_03). Default: all.")
    parser.add_argument("--calib", type=Path, default=DEFAULT_CALIB,
                        help="Path to the surfnav camera YAML config.")
    parser.add_argument("--output_dir", type=Path, default=default_save_path,
                        help="Directory to save output trajectory files.")
    parser.add_argument("--staging_dir", type=Path, default=default_staging,
                        help="Directory for per-trial staging dirs.")
    parser.add_argument("--config_path", type=Path, default=default_config,
                        help="Path to filter_config.yaml for depth/seg thresholds.")
    args = parser.parse_args()

    try:
        with open(args.config_path) as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: config file not found at {args.config_path}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.staging_dir.mkdir(parents=True, exist_ok=True)

    if args.trials:
        trial_dirs = sorted([args.trials_dir / t for t in args.trials])
    else:
        trial_dirs = sorted(args.trials_dir.glob('trial_*'))

    use_depth = args.filter_mode in ('depth', 'both')
    use_seg   = args.filter_mode in ('segmentation', 'both')

    print(f"Processing {len(trial_dirs)} trials using '{args.filter_mode}' mode.")

    for trial_dir in tqdm(trial_dirs, desc="Processing trials"):
        name = trial_dir.name

        staging_dir = prepare_staging(trial_dir, args.staging_dir, use_depth, use_seg)
        if staging_dir is None:
            continue

        output_traj = args.output_dir / f"campus_{name}_{args.filter_mode}_filtered_traj.txt"

        command = [
            "python", str(filter_script_path),
            "--input-path",    str(staging_dir),
            "--config_path",   str(args.calib),
            "--config_type",   "surfnav",
            "--image_dirname", "images",
            "--output_path",   str(output_traj),
        ]

        if args.filter_mode == 'depth':
            depth_cfg = config.get('depth', {})
            mask_path = staging_dir / 'depth_masks'
            command.extend([
                "depth",
                "--mask-path", str(mask_path),
                "--tl", str(depth_cfg.get('tl', 0.0)),
                "--th", str(depth_cfg.get('th', 50.0)),
            ])
            if depth_cfg.get('normalize', False):
                command.append("--normalize")

        elif args.filter_mode == 'segmentation':
            seg_cfg   = config.get('segmentation', {})
            mask_path = staging_dir / 'seg_masks'
            command.extend([
                "segmentation",
                "--mask-path",  str(mask_path),
                "--filter-ids", str(seg_cfg.get('filter_ids', '0,1,2,3,4,5,7,8,9')),
            ])

        elif args.filter_mode == 'both':
            depth_cfg  = config.get('depth', {})
            seg_cfg    = config.get('segmentation', {})
            command.extend([
                "both",
                "--depth-mask-path", str(staging_dir / 'depth_masks'),
                "--seg-mask-path",   str(staging_dir / 'seg_masks'),
                "--tl",              str(depth_cfg.get('tl', 0.0)),
                "--th",              str(depth_cfg.get('th', 50.0)),
                "--filter-ids",      str(seg_cfg.get('filter_ids', '0,1,2,3,4,5,7,8,9')),
            ])
            if depth_cfg.get('normalize', False):
                command.append("--normalize")

        tqdm.write(f"----- {name} ({args.filter_mode}) -----")
        try:
            subprocess.run(command, check=True, capture_output=False, text=True)
            tqdm.write(f"--- {name} successful ---")
        except subprocess.CalledProcessError as e:
            tqdm.write(f"--- Error on {name}: return code {e.returncode} ---")
        except Exception as e:
            tqdm.write(f"--- Unexpected error on {name}: {e} ---")

    print("All trials processed.")


if __name__ == "__main__":
    main()
