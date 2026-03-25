import numpy as np
'''
camera.py deals with handling camera intrinsics
'''
import yaml
from pathlib import Path
def parse_camera_surfnav(path:Path) -> dict:
    
    input_config = {}
    camera_config = {}
    with open(path, 'r') as file:
        input_config = yaml.safe_load(file)
        
    camera_config['model'] = 'OPENCV'
    
    width = 0
    height = 0.30
    cam_matrix = []
    dist_matrix = []
    try:
        width = input_config['image_width']
        height = input_config['image_height']
        cam_matrix = input_config['camera_matrix']['data']
        dist_matrix = input_config['distortion_coefficients']['data']
    
    except KeyError as e:
        print(f'Error trying to parse surfnav data: {e}')
        
    assert(len(cam_matrix) == 9)
    
    # extract intrinsics and distortion coefficients in the format poselib expects
    camera_data = [cam_matrix[0], cam_matrix[2], cam_matrix[4], cam_matrix[5]]
    camera_data.extend(dist_matrix)

    camera_config['width'] = width
    camera_config['height'] = height
    camera_config['params'] = camera_data 
    
    return camera_config


def parse_camera_euroc(path: Path) -> dict:
    camera_config = {}
    with open(path, 'r') as file:
        input_config = yaml.safe_load(file)

    camera_config['model'] = 'OPENCV'
    
    #verified from https://github.com/poselib/poselib/blob/main/python/poselib.cc#L292C25-L292C25
    #fx, fy, cx, cy
    camera_data = input_config['intrinsics']
    camera_data.extend(input_config['distortion_coefficients'])

    camera_config['width'] = input_config['resolution'][0]
    camera_config['height'] = input_config['resolution'][1]
    camera_config['params'] = camera_data
    
    return camera_config

def parse_camera_kitti(path: Path) -> dict:
    camera_config = {}
    with open(path, 'r') as file:
        for line in file:
            if line.startswith('P0:'):
                parts = line.split()[1:]
                p_matrix = np.array([float(x) for x in parts]).reshape(3,4)
                
                fx = p_matrix[0,0]
                fy = p_matrix[1,1]
                cx = p_matrix[0,2]
                cy = p_matrix[1,2]
                
                camera_config['model'] = 'OPENCV'
                camera_config['params'] = [fx, fy, cx, cy, 0, 0, 0, 0]
                
                return camera_config
                
    raise ValueError("Could not find P0 in calibration file")




 
    


    
    