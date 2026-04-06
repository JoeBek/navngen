"""
Wrapper script to run filter_trajectory.py on all nuScenes CAM_FRONT scenes,
mirroring the logic of run_kitti_filter.py.

For each scene this script:
  1. Builds a staging directory containing:
       images/       <- symlinks to the scene's CAM_FRONT frames (sorted by timestamp)
       times.txt     <- per-frame timestamps in seconds
       calib.yaml    <- euroc-format intrinsics for this scene
       depth_masks/  <- symlinks to depth .npy files for this scene  (depth mode)
       seg_masks/    <- symlinks to seg .npy files for this scene     (seg mode)
  2. Calls filter_trajectory.py via subprocess with --config_type euroc.

nuScenes images are undistorted, so distortion coefficients are set to zero.
"""

import argparse
import json
import subprocess
import sys

# Official nuScenes v1.0-trainval splits (700 train / 150 val / 150 test)
# Source: nutonomy/nuscenes-devkit nuscenes/utils/splits.py
NUSCENES_VAL_SCENES = {
    'scene-0003', 'scene-0012', 'scene-0013', 'scene-0014', 'scene-0015',
    'scene-0016', 'scene-0017', 'scene-0018', 'scene-0035', 'scene-0036',
    'scene-0038', 'scene-0039', 'scene-0092', 'scene-0093', 'scene-0094',
    'scene-0095', 'scene-0096', 'scene-0097', 'scene-0098', 'scene-0099',
    'scene-0100', 'scene-0101', 'scene-0102', 'scene-0103', 'scene-0104',
    'scene-0105', 'scene-0106', 'scene-0107', 'scene-0108', 'scene-0109',
    'scene-0110', 'scene-0221', 'scene-0268', 'scene-0269', 'scene-0270',
    'scene-0271', 'scene-0272', 'scene-0273', 'scene-0274', 'scene-0275',
    'scene-0276', 'scene-0277', 'scene-0278', 'scene-0329', 'scene-0330',
    'scene-0331', 'scene-0332', 'scene-0344', 'scene-0345', 'scene-0346',
    'scene-0519', 'scene-0520', 'scene-0521', 'scene-0522', 'scene-0523',
    'scene-0524', 'scene-0552', 'scene-0553', 'scene-0554', 'scene-0555',
    'scene-0556', 'scene-0557', 'scene-0558', 'scene-0559', 'scene-0560',
    'scene-0561', 'scene-0562', 'scene-0563', 'scene-0564', 'scene-0565',
    'scene-0625', 'scene-0626', 'scene-0627', 'scene-0629', 'scene-0630',
    'scene-0632', 'scene-0633', 'scene-0634', 'scene-0635', 'scene-0636',
    'scene-0637', 'scene-0638', 'scene-0770', 'scene-0771', 'scene-0775',
    'scene-0777', 'scene-0778', 'scene-0780', 'scene-0781', 'scene-0782',
    'scene-0783', 'scene-0784', 'scene-0794', 'scene-0795', 'scene-0796',
    'scene-0797', 'scene-0798', 'scene-0799', 'scene-0800', 'scene-0802',
    'scene-0904', 'scene-0905', 'scene-0906', 'scene-0907', 'scene-0908',
    'scene-0909', 'scene-0910', 'scene-0911', 'scene-0912', 'scene-0913',
    'scene-0914', 'scene-0915', 'scene-0916', 'scene-0917', 'scene-0919',
    'scene-0920', 'scene-0921', 'scene-0922', 'scene-0923', 'scene-0924',
    'scene-0925', 'scene-0926', 'scene-0927', 'scene-0928', 'scene-0929',
    'scene-0930', 'scene-0931', 'scene-0962', 'scene-0963', 'scene-0966',
    'scene-0967', 'scene-0968', 'scene-0969', 'scene-0971', 'scene-0972',
    'scene-1059', 'scene-1060', 'scene-1061', 'scene-1062', 'scene-1063',
    'scene-1064', 'scene-1065', 'scene-1066', 'scene-1067', 'scene-1068',
    'scene-1069', 'scene-1070', 'scene-1071', 'scene-1072', 'scene-1073',
}

NUSCENES_TEST_SCENES = {
    'scene-0077', 'scene-0078', 'scene-0079', 'scene-0080', 'scene-0081',
    'scene-0082', 'scene-0083', 'scene-0084', 'scene-0085', 'scene-0086',
    'scene-0087', 'scene-0088', 'scene-0089', 'scene-0090', 'scene-0091',
    'scene-0111', 'scene-0112', 'scene-0113', 'scene-0114', 'scene-0115',
    'scene-0116', 'scene-0117', 'scene-0118', 'scene-0119', 'scene-0140',
    'scene-0142', 'scene-0143', 'scene-0144', 'scene-0145', 'scene-0146',
    'scene-0147', 'scene-0148', 'scene-0265', 'scene-0266', 'scene-0279',
    'scene-0280', 'scene-0281', 'scene-0282', 'scene-0307', 'scene-0308',
    'scene-0309', 'scene-0310', 'scene-0311', 'scene-0312', 'scene-0313',
    'scene-0314', 'scene-0333', 'scene-0334', 'scene-0335', 'scene-0336',
    'scene-0337', 'scene-0338', 'scene-0339', 'scene-0340', 'scene-0341',
    'scene-0342', 'scene-0343', 'scene-0481', 'scene-0482', 'scene-0483',
    'scene-0484', 'scene-0485', 'scene-0486', 'scene-0487', 'scene-0488',
    'scene-0489', 'scene-0490', 'scene-0491', 'scene-0492', 'scene-0493',
    'scene-0494', 'scene-0495', 'scene-0496', 'scene-0497', 'scene-0498',
    'scene-0547', 'scene-0548', 'scene-0549', 'scene-0550', 'scene-0551',
    'scene-0601', 'scene-0602', 'scene-0603', 'scene-0604', 'scene-0606',
    'scene-0607', 'scene-0608', 'scene-0609', 'scene-0610', 'scene-0611',
    'scene-0612', 'scene-0613', 'scene-0614', 'scene-0615', 'scene-0616',
    'scene-0617', 'scene-0618', 'scene-0619', 'scene-0620', 'scene-0621',
    'scene-0622', 'scene-0623', 'scene-0624', 'scene-0827', 'scene-0828',
    'scene-0829', 'scene-0830', 'scene-0831', 'scene-0833', 'scene-0834',
    'scene-0835', 'scene-0836', 'scene-0837', 'scene-0838', 'scene-0839',
    'scene-0840', 'scene-0841', 'scene-0842', 'scene-0844', 'scene-0845',
    'scene-0846', 'scene-0932', 'scene-0933', 'scene-0935', 'scene-0936',
    'scene-0937', 'scene-0938', 'scene-0939', 'scene-0940', 'scene-0941',
    'scene-0942', 'scene-0943', 'scene-1026', 'scene-1027', 'scene-1028',
    'scene-1029', 'scene-1030', 'scene-1031', 'scene-1032', 'scene-1033',
    'scene-1034', 'scene-1035', 'scene-1036', 'scene-1037', 'scene-1038',
    'scene-1039', 'scene-1040', 'scene-1041', 'scene-1042', 'scene-1043',
}
import yaml
from pathlib import Path
from tqdm import tqdm


# ---------------------------------------------------------------------------
# nuScenes metadata helpers
# ---------------------------------------------------------------------------

def load_json(path):
    with open(path) as f:
        return json.load(f)


def build_scene_data(dataroot: Path, include_sweeps: bool):
    """
    Returns a list of scene dicts, each with:
        name        : str  (e.g. 'scene-0001')
        frames      : list of sample_data dicts, sorted by timestamp
        K           : 3x3 intrinsics list [[fx,0,cx],[0,fy,cy],[0,0,1]]
        width, height: int
    """
    meta = dataroot / 'v1.0-trainval'

    sensors    = load_json(meta / 'sensor.json')
    calibrated = load_json(meta / 'calibrated_sensor.json')
    sd_list    = load_json(meta / 'sample_data.json')
    samples    = load_json(meta / 'sample.json')
    scenes     = load_json(meta / 'scene.json')

    sensor_map  = {s['token']: s for s in sensors}
    cal_map     = {c['token']: c for c in calibrated}
    sample_map  = {s['token']: s for s in samples}
    scene_map   = {s['token']: s for s in scenes}
    sd_map      = {s['token']: s for s in sd_list}

    front_cal_tokens = {
        c['token'] for c in calibrated
        if sensor_map.get(c.get('sensor_token', ''), {}).get('channel') == 'CAM_FRONT'
    }

    # Find chain roots (first frame of each scene's CAM_FRONT stream)
    roots = [
        sd for sd in sd_list
        if sd['calibrated_sensor_token'] in front_cal_tokens and sd['prev'] == ''
    ]

    scene_data = []
    for root in roots:
        # Walk the linked list to collect all frames for this scene
        frames = []
        node = root
        while node is not None:
            if include_sweeps or node['is_key_frame']:
                frames.append(node)
            next_token = node.get('next', '')
            node = sd_map.get(next_token) if next_token else None

        if not frames:
            continue

        # Resolve scene name
        sample  = sample_map.get(root['sample_token'], {})
        scene   = scene_map.get(sample.get('scene_token', ''), {})
        name    = scene.get('name', root['token'])

        # Intrinsics from the first frame's calibration
        cal = cal_map[root['calibrated_sensor_token']]
        K   = cal['camera_intrinsic']  # [[fx,0,cx],[0,fy,cy],[0,0,1]]

        scene_data.append({
            'name': name,
            'frames': sorted(frames, key=lambda s: s['timestamp']),
            'K': K,
            'width': root['width'],
            'height': root['height'],
        })

    return scene_data


# ---------------------------------------------------------------------------
# Staging directory helpers
# ---------------------------------------------------------------------------

def prepare_staging(scene: dict, dataroot: Path, staging_root: Path,
                    depth_masks_dir: Path | None, seg_masks_dir: Path | None):
    """
    Creates (or refreshes) a staging directory for one scene.
    Returns the staging Path, or None if a required mask directory is missing.
    """
    staging_dir = staging_root / scene['name']
    img_dir     = staging_dir / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)

    frames = scene['frames']

    # --- image symlinks ---
    for sd in frames:
        src  = (dataroot / sd['filename']).resolve()
        dst  = img_dir / src.name
        if not dst.exists():
            dst.symlink_to(src)

    # --- times.txt (timestamps in seconds) ---
    times_path = staging_dir / 'times.txt'
    with open(times_path, 'w') as f:
        for sd in frames:
            f.write(f"{sd['timestamp'] / 1e6:.6f}\n")

    # --- calib.yaml (euroc format, no distortion since nuScenes is undistorted) ---
    K = scene['K']
    calib = {
        'intrinsics': [K[0][0], K[1][1], K[0][2], K[1][2]],  # fx, fy, cx, cy
        'distortion_coefficients': [0.0, 0.0, 0.0, 0.0],
        'resolution': [scene['width'], scene['height']],
    }
    calib_path = staging_dir / 'calib.yaml'
    with open(calib_path, 'w') as f:
        yaml.dump(calib, f)

    stems = {(dataroot / sd['filename']).stem for sd in frames}

    # --- depth mask symlinks ---
    if depth_masks_dir is not None:
        dmask_dir = staging_dir / 'depth_masks'
        dmask_dir.mkdir(exist_ok=True)
        missing = 0
        for stem in stems:
            src = (depth_masks_dir / f"{stem}.npy").resolve()
            dst = dmask_dir / f"{stem}.npy"
            if src.exists():
                if not dst.is_symlink():
                    dst.symlink_to(src)
            else:
                missing += 1
        if missing:
            tqdm.write(f"  [{scene['name']}] WARNING: {missing} depth masks missing.")

    # --- seg mask symlinks ---
    if seg_masks_dir is not None:
        smask_dir = staging_dir / 'seg_masks'
        smask_dir.mkdir(exist_ok=True)
        missing = 0
        for stem in stems:
            src = (seg_masks_dir / f"{stem}.npy").resolve()
            dst = smask_dir / f"{stem}.npy"
            if src.exists():
                if not dst.is_symlink():
                    dst.symlink_to(src)
            else:
                missing += 1
        if missing:
            tqdm.write(f"  [{scene['name']}] WARNING: {missing} seg masks missing.")

    return staging_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    project_root       = Path(__file__).resolve().parents[1]
    filter_script_path = project_root / 'scripts' / 'filter_trajectory.py'
    default_save_path  = project_root / 'assets' / 'outputs' / 'nuscenes'
    default_config     = project_root / 'scripts' / 'configs' / 'filter_config.yaml'

    if not filter_script_path.exists():
        print(f"Error: filter_trajectory.py not found at {filter_script_path}")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Run filtering for nuScenes CAM_FRONT scenes.")
    parser.add_argument("--filter-mode", type=str, choices=['depth', 'segmentation', 'both'], default='depth',
                        help="The type of filtering to apply. 'both' applies depth then segmentation on the same keypoints.")
    parser.add_argument("--dataroot", type=Path, default=Path("/home/joe/data/nuscenes"),
                        help="Path to the nuScenes dataset root.")
    parser.add_argument("--depth_masks_dir", type=Path,
                        default=Path("/home/joe/data/nuscenes") / "depth" / "depth_masks",
                        help="Directory containing depth .npy masks (output of nuscenes_depth.py).")
    parser.add_argument("--seg_masks_dir", type=Path,
                        default=Path("/home/joe/data/nuscenes") / "seg" / "seg_masks",
                        help="Directory containing seg .npy masks (output of nuscenes_seg.py).")
    parser.add_argument("--output_dir", type=Path, default=default_save_path,
                        help="Directory to save output trajectory files.")
    parser.add_argument("--staging_dir", type=Path,
                        default=Path("/home/joe/data/nuscenes") / "staging",
                        help="Directory for per-scene staging dirs (images, times.txt, calib.yaml, mask symlinks).")
    parser.add_argument("--config_path", type=Path, default=default_config,
                        help="Path to the filter_config.yaml for depth/seg thresholds.")
    parser.add_argument("--sweeps", action="store_true",
                        help="Include sweep frames (not just key frames).")
    parser.add_argument("--scenes", type=str, default=None,
                        help="Comma-separated list of scene names to process (e.g. 'scene-0001,scene-0005'). "
                             "Defaults to all scenes.")
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument("--val", action="store_true",
                             help="Process only the 150 official nuScenes validation scenes.")
    split_group.add_argument("--test", action="store_true",
                             help="Process only the 150 official nuScenes test scenes.")
    args = parser.parse_args()

    try:
        with open(args.config_path) as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: config file not found at {args.config_path}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.staging_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading nuScenes metadata from '{args.dataroot}'...")
    all_scenes = build_scene_data(args.dataroot, include_sweeps=args.sweeps)

    if args.val:
        all_scenes = [s for s in all_scenes if s['name'] in NUSCENES_VAL_SCENES]
    elif args.test:
        all_scenes = [s for s in all_scenes if s['name'] in NUSCENES_TEST_SCENES]
    elif args.scenes:
        requested = set(args.scenes.split(','))
        all_scenes = [s for s in all_scenes if s['name'] in requested]

    print(f"Processing {len(all_scenes)} scenes using '{args.filter_mode}' mode.")

    depth_masks_dir = args.depth_masks_dir if args.filter_mode in ('depth', 'both') else None
    seg_masks_dir   = args.seg_masks_dir   if args.filter_mode in ('segmentation', 'both') else None

    for scene in tqdm(all_scenes, desc="Processing scenes"):
        name = scene['name']

        # --- Prepare staging directory ---
        staging_dir = prepare_staging(
            scene, args.dataroot, args.staging_dir,
            depth_masks_dir, seg_masks_dir
        )

        output_traj = args.output_dir / f"nuscenes_{name}_{args.filter_mode}_filtered_traj.txt"

        # --- Build filter_trajectory.py command ---
        command = [
            "python", str(filter_script_path),
            "--input-path",   str(staging_dir),
            "--config_path",  str(staging_dir / 'calib.yaml'),
            "--config_type",  "euroc",
            "--image_dirname", "images",
            "--output_path",  str(output_traj),
        ]

        if args.filter_mode == 'depth':
            depth_cfg = config.get('depth', {})
            mask_path = staging_dir / 'depth_masks'
            if not mask_path.exists() or not any(mask_path.iterdir()):
                tqdm.write(f"[{name}] No depth masks found, skipping.")
                continue
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
            if not mask_path.exists() or not any(mask_path.iterdir()):
                tqdm.write(f"[{name}] No seg masks found, skipping.")
                continue
            command.extend([
                "segmentation",
                "--mask-path",  str(mask_path),
                "--filter-ids", str(seg_cfg.get('filter_ids', '0,1,2,3,4,5,7,8,9')),
            ])

        elif args.filter_mode == 'both':
            depth_cfg  = config.get('depth', {})
            seg_cfg    = config.get('segmentation', {})
            dmask_path = staging_dir / 'depth_masks'
            smask_path = staging_dir / 'seg_masks'
            if not dmask_path.exists() or not any(dmask_path.iterdir()):
                tqdm.write(f"[{name}] No depth masks found, skipping.")
                continue
            if not smask_path.exists() or not any(smask_path.iterdir()):
                tqdm.write(f"[{name}] No seg masks found, skipping.")
                continue
            command.extend([
                "both",
                "--depth-mask-path", str(dmask_path),
                "--seg-mask-path",   str(smask_path),
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

    print("All scenes processed.")


if __name__ == "__main__":
    main()
