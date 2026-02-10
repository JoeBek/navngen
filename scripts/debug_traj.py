
import sys
from pathlib import Path

# Add the project root to sys.path
# This assumes the script is run from within the project structure
project_root = Path(__file__).resolve().parents[1] # Go up one level from debug_traj.py to /home/joe/vt/research/glue/navngen/
sys.path.insert(0, str(project_root))

from src.navngen.export_trajectory import load_frames
from src.navngen.frame import Frame # Ensure src.navngen.frame is loaded

if __name__=="__main__":
    # Patch for pickle to find the 'frame' module
    if 'src.navngen.frame' in sys.modules and 'frame' not in sys.modules:
        sys.modules['frame'] = sys.modules['src.navngen.frame']
    
    frame_path = Path(__file__).resolve().parent.parent / "assets" / "outputs" / "01_pickle.pkl.gz"
    
    frames = load_frames(frame_path)

    for frame in frames:
        print(frame.timestamp)