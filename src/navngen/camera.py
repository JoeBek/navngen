
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
    


    
    