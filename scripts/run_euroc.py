import argparse
from pathlib import Path
import subprocess
import sys
from tqdm import tqdm

# category prefix → subdirectory under euroc root
SEQUENCE_CATEGORIES = {
    'MH': 'machine_hall',
    'V1': 'vicon_room1',
    'V2': 'vicon_room2',
}


def find_mav0_dirs(euroc_root: Path):
    """Return sorted list of mav0 Path objects found under euroc_root/*/*/mav0."""
    return sorted(euroc_root.glob('*/*/mav0'))


def main():
    project_root = Path(__file__).resolve().parents[1]
    get_trajectory_script = project_root / 'scripts' / 'get_trajectory.py'
    default_output_dir = project_root / 'assets' / 'outputs' / 'euroc'

    parser = argparse.ArgumentParser(
        description="Run navngen on all EuRoC sequences.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--euroc_data_path", type=Path, default=Path("/home/joe/data/euroc"),
        help="Path to the root EuRoC data directory.",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=default_output_dir,
        help="Directory to save output trajectory files.",
    )
    parser.add_argument(
        "--sequences", type=str, default=None,
        help="Comma-separated list of sequence names to run (e.g. MH_01_easy,V1_01_easy).\n"
             "If omitted, all sequences found are processed.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    mav0_dirs = find_mav0_dirs(args.euroc_data_path)
    if not mav0_dirs:
        print(f"No mav0 directories found under {args.euroc_data_path}")
        sys.exit(1)

    if args.sequences:
        requested = set(args.sequences.split(','))
        mav0_dirs = [d for d in mav0_dirs if d.parent.name in requested]
        if not mav0_dirs:
            print(f"None of the requested sequences were found: {requested}")
            sys.exit(1)

    print(f"Processing {len(mav0_dirs)} sequence(s).")

    for mav0_path in tqdm(mav0_dirs, desc="EuRoC sequences"):
        seq_name = mav0_path.parent.name  # e.g. MH_01_easy
        config_path = mav0_path / 'cam0' / 'sensor.yaml'
        output_path = args.output_dir / f"{seq_name}_traj.txt"

        if not config_path.exists():
            tqdm.write(f"[SKIP] {seq_name}: sensor.yaml not found at {config_path}")
            continue

        command = [
            sys.executable, str(get_trajectory_script),
            "--input_path",  str(mav0_path),
            "--config_path", str(config_path),
            "--config_type", "euroc",
            "--output_path", str(output_path),
        ]

        tqdm.write(f"----- {seq_name} -----")
        try:
            subprocess.run(command, check=True)
            tqdm.write(f"[OK] {seq_name} → {output_path}")
        except subprocess.CalledProcessError as e:
            tqdm.write(f"[ERROR] {seq_name}: return code {e.returncode}")
        except Exception as e:
            tqdm.write(f"[ERROR] {seq_name}: {e}")

    print("All sequences processed.")


if __name__ == "__main__":
    main()
