In nuScenes, "ground truth" can refer to several things: the vehicle's trajectory (Ego Pose), the 3D bounding boxes (Annotations), or the sensor calibrations (Extrinsics/Intrinsics).

The most efficient way to access these is using the nuscenes-devkit.
1. Setup and Initialization

First, ensure you have the devkit installed and the dataset initialized.
Python

# pip install nuscenes-devkit
from nuscenes.nuscenes import NuScenes

nusc = NuScenes(version='v1.0-mini', dataroot='/path/to/data', verbose=True)

2. Getting Camera Intrinsics

Intrinsics are stored in the calibrated_sensor table. Each sample_data (like an image) points to a specific calibration record because the intrinsics can technically change if a sensor is replaced or recalibrated between logs.
Python

# Get a sample and a specific camera's data token
my_sample = nusc.sample[0]
cam_front_token = my_sample['data']['CAM_FRONT']
cam_data = nusc.get('sample_data', cam_front_token)

# Get the calibration record
calib_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
intrinsics = calib_record['camera_intrinsic']

print(f"Intrinsics Matrix:\n{intrinsics}")

    Note: nuScenes images come undistorted and rectified, so you generally don't need to worry about distortion coefficients.

3. Getting Ground Truth (GT)
Vehicle Trajectory (Ego Pose)

The ego_pose contains the global position and orientation of the vehicle at the time the sensor data was captured.
Python

ego_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])
translation = ego_pose['translation'] # [x, y, z] in meters
rotation = ego_pose['rotation']       # [w, x, y, z] quaternion

3.D Object Annotations

If you need the actual labels (bounding boxes) for a specific sample:
Python

# Get all annotation tokens for a sample
ann_tokens = my_sample['anns']

for token in ann_tokens:
    ann = nusc.get('sample_annotation', token)
    print(f"Category: {ann['category_name']}, Box: {ann['translation']}")

4. Summary Table of GT Components
Data Type	Table Name	Key Fields
Intrinsics	calibrated_sensor	camera_intrinsic
Extrinsics	calibrated_sensor	translation, rotation (relative to ego)
Trajectory	ego_pose	translation, rotation (global frame)
Objects	sample_annotation	category_name, size, translation

Since you've been working with mmsegmentation and KITTI, you might find that nuScenes uses a different coordinate system (Right-Handed, Z-up). Would you like me to show you how to convert these poses into the KITTI format?
