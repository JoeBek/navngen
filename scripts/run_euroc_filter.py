import argparse
from pathlib import Path
import subprocess
import sys
import yaml
from tqdm import tqdm

SEQUENCE_CATEGORIES = {
    'MH': 'machine_hall',
    'V1': 'vicon_room1',
    'V2': 'vicon_room2',
}


def find_mav0_dirs(euroc_root: Path):
    return sorted(euroc_root.glob('*/*/mav0'))


def main():
    project_root = Path(__file__).resolve().parents[1]
    filter_script = project_root / 'scripts' / 'filter_trajectory.py'
    default_output_dir = project_root / 'assets' / 'outputs' / 'euroc'
    default_config_path = project_root / 'scripts' / 'configs' / 'filter_config.yaml'

    parser = argparse.ArgumentParser(
        description='Run filter_trajectory.py on all EuRoC sequences.',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--filter-mode', type=str,
                        choices=['depth', 'segmentation', 'both'], default='depth',
                        help='Filtering mode.')
    parser.add_argument('--euroc_data_path', type=Path, default=Path('/home/joe/data/euroc'),
                        help='Root EuRoC data directory.')
    parser.add_argument('--output_dir', type=Path, default=default_output_dir,
                        help='Directory to save output trajectory files.')
    parser.add_argument('--config_path', type=Path, default=default_config_path,
                        help='Path to filter_config.yaml.')
    parser.add_argument('--sequences', type=str, default=None,
                        help='Comma-separated sequence names to process.\n'
                             'Default: all sequences found.')
    args = parser.parse_args()

    try:
        with open(args.config_path) as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f'Error: config not found at {args.config_path}')
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    mav0_dirs = find_mav0_dirs(args.euroc_data_path)
    if args.sequences:
        requested = set(args.sequences.split(','))
        mav0_dirs = [d for d in mav0_dirs if d.parent.name in requested]
    if not mav0_dirs:
        print(f'No sequences found under {args.euroc_data_path}')
        sys.exit(1)

    print(f'Processing {len(mav0_dirs)} sequence(s) with mode "{args.filter_mode}".')

    for mav0_path in tqdm(mav0_dirs, desc='EuRoC filter'):
        seq_name = mav0_path.parent.name
        config_path = mav0_path / 'cam0' / 'sensor.yaml'
        output_path = args.output_dir / f'{seq_name}_{args.filter_mode}_filtered_traj.txt'

        if not config_path.exists():
            tqdm.write(f'[SKIP] {seq_name}: sensor.yaml not found.')
            continue

        # Mask directories follow the same layout produced by euroc_depth.py / euroc_seg.py
        depth_mask_path = args.euroc_data_path / 'depth' / seq_name / 'masks'
        seg_mask_path   = args.euroc_data_path / 'seg'   / seq_name / 'masks'

        command = [
            sys.executable, str(filter_script),
            '--input-path',  str(mav0_path),
            '--config_path', str(config_path),
            '--config_type', 'euroc',
            '--output_path', str(output_path),
        ]

        if args.filter_mode == 'depth':
            if not depth_mask_path.exists():
                tqdm.write(f'[SKIP] {seq_name}: depth masks not found at {depth_mask_path}')
                continue
            depth_cfg = config.get('depth', {})
            command += [
                'depth',
                '--mask-path', str(depth_mask_path),
                '--tl', str(depth_cfg.get('tl', 0.0)),
                '--th', str(depth_cfg.get('th', 50.0)),
            ]
            if depth_cfg.get('normalize', False):
                command.append('--normalize')

        elif args.filter_mode == 'segmentation':
            if not seg_mask_path.exists():
                tqdm.write(f'[SKIP] {seq_name}: seg masks not found at {seg_mask_path}')
                continue
            seg_cfg = config.get('segmentation', {})
            command += [
                'segmentation',
                '--mask-path', str(seg_mask_path),
                '--filter-ids', str(seg_cfg.get('filter_ids', '')),
            ]

        elif args.filter_mode == 'both':
            if not depth_mask_path.exists():
                tqdm.write(f'[SKIP] {seq_name}: depth masks not found at {depth_mask_path}')
                continue
            if not seg_mask_path.exists():
                tqdm.write(f'[SKIP] {seq_name}: seg masks not found at {seg_mask_path}')
                continue
            depth_cfg = config.get('depth', {})
            seg_cfg   = config.get('segmentation', {})
            command += [
                'both',
                '--depth-mask-path', str(depth_mask_path),
                '--seg-mask-path',   str(seg_mask_path),
                '--tl',              str(depth_cfg.get('tl', 0.0)),
                '--th',              str(depth_cfg.get('th', 50.0)),
                '--filter-ids',      str(seg_cfg.get('filter_ids', '')),
            ]
            if depth_cfg.get('normalize', False):
                command.append('--normalize')

        tqdm.write(f'----- {seq_name} ({args.filter_mode}) -----')
        try:
            subprocess.run(command, check=True)
            tqdm.write(f'[OK] {seq_name} → {output_path}')
        except subprocess.CalledProcessError as e:
            tqdm.write(f'[ERROR] {seq_name}: return code {e.returncode}')
        except Exception as e:
            tqdm.write(f'[ERROR] {seq_name}: {e}')

    print('All sequences processed.')


if __name__ == '__main__':
    main()
