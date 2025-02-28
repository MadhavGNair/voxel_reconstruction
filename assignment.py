import glm
import random
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from voxel_reconstruction import VoxelReconstructor

block_size = 1.0

def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -block_size, z*block_size - depth/2])
            colors.append([1.0, 1.0, 1.0] if (x+z) % 2 == 0 else [0, 0, 0])
    return data, colors

def set_voxel_positions(width, height, depth):
    # Generates voxel locations from reconstruction and adds grid edges
    print(width, height, depth)
    vr = VoxelReconstructor(width, height, depth, (644, 486))
    data, colors = [], []
    
    # Add voxels from reconstruction
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                if vr.voxel_space[x, y, z, 0] >= 4:  # If all 4 cameras see this voxel
                    data.append([x*block_size - width/2, -(z*block_size - depth/2), y*block_size])
                    camera_count = vr.voxel_space[x, y, z, 0]
                    # r = vr.voxel_space[x, y, z, 1] / camera_count
                    # g = vr.voxel_space[x, y, z, 2] / camera_count
                    # b = vr.voxel_space[x, y, z, 3] / camera_count
                    
                    # Normalize to [0,1] range for rendering
                    # colors.append([r/255.0, g/255.0, b/255.0])
                    # Create a gradient based on the voxel position
                    r = x / width  # Red varies with x position
                    g = y / height  # Green varies with y position
                    b = z / depth   # Blue varies with z position
                    colors.append([r, g, b])
    
    return data, colors

def load_camera_parameters(camera_data_path):
    try:
        tree = ET.parse(camera_data_path)
        root = tree.getroot()

        camera_matrix = np.fromstring(root.find('CameraMatrix/data').text.replace('\n', ' '), sep=' ').reshape(3, 3)
        dist_coeffs = np.fromstring(root.find('DistortionCoeffs/data').text.replace('\n', ' '), sep=' ')
        rvec = np.fromstring(root.find('RotationVector/data').text.replace('\n', ' '), sep=' ')
        tvec = np.fromstring(root.find('TranslationVector/data').text.replace('\n', ' '), sep=' ')
        return camera_matrix, dist_coeffs, rvec, tvec
    except FileNotFoundError:
        print(f"Error: XML file not found at {camera_data_path}")
        return None, None, None, None
    except Exception as e:
        print(f"Error parsing XML: {e}")
        return None, None, None, None

def get_cam_positions():
    cam_positions = []
    scale = block_size / 115
    for i in range(1, 5):
        camera_data_path = f"./data/cam{i}/config.xml"
        camera_matrix, dist_coeffs, rvec, tvec = load_camera_parameters(camera_data_path)
        R, _ = cv2.Rodrigues(rvec)
        position = -np.matrix(R).T * np.matrix(tvec).T * scale
        # OpenGL uses Y-up, but the camera uses Z-up, so we need to flip the Y-axis
        cam_positions.append([position[0], position[2], position[1]])

    cam_colors = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]
    return cam_positions, cam_colors


def get_cam_rotation_matrices():
    cam_rotations = []
    for i in range(1, 5):
        camera_data_path = f"./data/cam{i}/config.xml"
        camera_matrix, dist_coeffs, rvec, tvec = load_camera_parameters(camera_data_path)
        
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        cam_rotation = np.identity(4)

        for row in range(3):
            for col in range(3):
                cam_rotation[row][col] = rotation_matrix[row][col]

        cam_rotation = cam_rotation.T

        flattened_matrix = cam_rotation.flatten()
        # rotate the camera 90 degrees around the Z-axis for correct viewing direction
        cam_rotations.append(glm.mat4(*flattened_matrix) * glm.rotate(np.pi/2, glm.vec3(0, 0, 1)))
    return cam_rotations


