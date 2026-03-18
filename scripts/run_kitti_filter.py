import argparse
from pathlib import Path
import subprocess
import yaml
import sys
from tqdm import tqdm

def main():
    """
    A wrapper script to run filter_trajectory.py on multiple KITTI sequences
    using either depth or segmentation filtering.
    """
    # Assuming this script is in src/navngen/
    project_root = Path(__file__).resolve().parents[1]
    filter_script_path = project_root / 'scripts' / 'filter_trajectory.py'
    default_save_path = project_root / 'assets' / 'outputs' / 'kitti'

    if not filter_script_path.exists():
        print(f"Error: filter_trajectory.py not found at {filter_script_path}")
        sys.exit(1)
        
    default_config_path = project_root / 'scripts' / 'configs' / 'filter_config.yaml'

    parser = argparse.ArgumentParser(description="Run filtering for KITTI sequences.")
    parser.add_argument("--filter-mode", type=str, choices=['depth', 'segmentation'], default='depth',
                        help="The type of filtering to apply.")
    parser.add_argument("--kitti_data_path", type=Path, default=Path("/home/joe/data/kitti"),
                        help="Path to the main KITTI data directory.")
    parser.add_argument("--output_dir", type=Path, default=default_save_path,
                        help="Directory to save the output trajectory files.")
    parser.add_argument("--config_path", type=Path, default=default_config_path,
                        help="Path to the YAML config file for filter parameters.")
    parser.add_argument("--start_trial", type=int, default=0, help="Starting KITTI trial number (0-21).")
    parser.add_argument("--end_trial", type=int, default=21, help="Ending KITTI trial number (0-21).")
    
    args = parser.parse_args()

    # Load configuration from YAML
    try:
        with open(args.config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: YAML config file not found at {args.config_path}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing KITTI trials {args.start_trial} to {args.end_trial} using '{args.filter_mode}' mode.")
    print(f"Using filter script: {filter_script_path}")
    print(f"Using config: {args.config_path}")

    # Use tqdm for the main loop
    for trial_num in tqdm(range(args.start_trial, args.end_trial + 1), desc="Processing Trials"):
        trial_str = f"{trial_num:02d}"
        
        # --- Construct common paths ---
        sequence_path = args.kitti_data_path / 'color' / 'dataset' / 'sequences' / trial_str
        calib_path = sequence_path / 'calib.txt'
        output_traj_path = args.output_dir / f"kitti_{trial_str}_{args.filter_mode}_filtered_traj.txt"

        # --- Check if necessary common paths exist ---
        if not sequence_path.exists() or not calib_path.exists():
            tqdm.write(f"Sequence or calibration path not found for trial {trial_str}, skipping.")
            continue

        # --- Build mode-specific command ---
        command = [
            "python", str(filter_script_path),
            "--input-path", str(sequence_path),
            "--config_path", str(calib_path),
            "--output_path", str(output_traj_path),
        ]

        if args.filter_mode == 'depth':
            depth_config = config.get('depth', {})
            tl = depth_config.get('tl', 0.0)
            th = depth_config.get('th', 50.0)
            normalize = depth_config.get('normalize', False)
            
            mask_path = args.kitti_data_path / 'depth' / trial_str / 'masks'
            if not mask_path.exists():
                tqdm.write(f"Depth data path not found for trial {trial_str}, skipping: {mask_path}")
                continue

            command.extend([
                "depth",
                "--mask-path", str(mask_path),
                "--tl", str(tl),
                "--th", str(th)
            ])
            if normalize:
                command.append("--normalize")

        elif args.filter_mode == 'segmentation':
            seg_config = config.get('segmentation', {})
            filter_ids = seg_config.get('filter_ids', "")
            
            mask_path = args.kitti_data_path / 'seg' / trial_str / 'masks'
            if not mask_path.exists():
                tqdm.write(f"Segmentation mask path not found for trial {trial_str}, skipping: {mask_path}")
                continue

            command.extend([
                "segmentation",
                "--mask-path", str(mask_path),
                "--filter-ids", str(filter_ids)
            ])
        
        tqdm.write(f"----- Running Trial {trial_str} ({args.filter_mode}) -----")
        # Use a more readable format for printing the command
        #tqdm.write("Executing: " + " ".join(f'"{c}"' if " " in c else c for c in command))
        
        # --- Run the command ---
        try:
            # Using capture_output=True will wait for the command to complete.
            # For long-running processes, consider streaming output if real-time feedback is needed.
            result = subprocess.run(command, check=True, capture_output=False, text=True)
            tqdm.write(f"--- Trial {trial_str} Successful ---")
            # To avoid flooding the console, we can write output to a log file instead
            # For now, let's keep it concise.
            tqdm.write("Output:" + result.stdout)
        except subprocess.CalledProcessError as e:
            tqdm.write(f"--- Error running trial {trial_str} ---")
            tqdm.write(f"Return Code: {e.returncode}")
        except Exception as e:
            tqdm.write(f"An unexpected error occurred during trial {trial_str}: {e}")

    print("All specified trials processed.")

if __name__ == "__main__":
    main()
