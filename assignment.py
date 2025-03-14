import random
import xml.etree.ElementTree as ET

import cv2
import glm
import numpy as np

from voxel_reconstruction import VoxelReconstructor

block_size = 1.0


def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append(
                [x * block_size - width / 2, -block_size, z * block_size - depth / 2]
            )
            colors.append([1.0, 1.0, 1.0] if (x + z) % 2 == 0 else [0, 0, 0])
    return data, colors


def set_voxel_positions(width, height, depth):
    visualize_mesh = False
    # generates voxel locations from reconstruction and adds grid edges
    reconstructor = VoxelReconstructor(width, height, depth, (644, 486))
    voxel_space, visibility_map, color_map, depth_map = reconstructor.reconstruct(save=True)
    data, colors = [], []

    # first pass: collect voxels and their colors with visibility information
    voxel_positions = []
    voxel_colors = []
    voxels_without_visibility = []

    print("First pass: calculating voxel positions and colors")
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                if voxel_space[x, y, z, 0] >= 4:
                    position = [
                        x * block_size - width / 2,
                        -(z * block_size - depth / 2),
                        y * block_size,
                    ]

                    # get color information with visibility weighting
                    visible_cameras = 0
                    weighted_color = np.zeros(3, dtype=np.float32)

                    # calculate weights based on visibility and distance
                    total_weight = 0
                    for cam_id in range(4):
                        if visibility_map[cam_id, x, y, z]:
                            # this camera has clear visibility to this voxel
                            visible_cameras += 1

                            # use inverse distance as weight (closer cameras have more influence)
                            distance = depth_map[cam_id, x, y, z]
                            if distance > 0:
                                weight = 1.0 / distance
                            else:
                                weight = 1.0

                            # get color from this camera's view
                            cam_color = (
                                color_map[cam_id, x, y, z].astype(np.float32) / 255.0
                            )

                            # add weighted contribution
                            weighted_color += cam_color * weight
                            total_weight += weight

                    # store voxel position and index
                    voxel_positions.append((x, y, z))

                    # if we have visible cameras, use weighted color
                    if visible_cameras > 0 and total_weight > 0:
                        # normalize by total weight
                        final_color = weighted_color / total_weight
                        voxel_colors.append(final_color)
                    else:
                        # mark this voxel for second pass coloring
                        voxels_without_visibility.append(len(voxel_positions) - 1)
                        # set temporary colors
                        camera_count = voxel_space[x, y, z, 0]
                        r = voxel_space[x, y, z, 1] / camera_count / 255.0
                        g = voxel_space[x, y, z, 2] / camera_count / 255.0
                        b = voxel_space[x, y, z, 3] / camera_count / 255.0
                        voxel_colors.append([r, g, b])

    # second pass: color voxels without visibility based on neighbors
    print("Second pass: coloring voxels without visibility")
    for idx in voxels_without_visibility:
        x, y, z = voxel_positions[idx]

        # find neighboring voxels with visibility information
        neighbor_colors = []
        neighbor_weights = []

        # check 26 neighbors (3x3x3 cube centered at current voxel)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    # skip the center voxel
                    if dx == 0 and dy == 0 and dz == 0:
                        continue

                    nx, ny, nz = x + dx, y + dy, z + dz

                    # check if neighbor is within bounds
                    if 0 <= nx < width and 0 <= ny < height and 0 <= nz < depth:
                        # find this neighbor in our voxel positions list
                        try:
                            neighbor_idx = voxel_positions.index((nx, ny, nz))
                            # only use neighbors that aren't in the voxels_without_visibility list
                            if neighbor_idx not in voxels_without_visibility:
                                # weight by inverse distance (diagonal neighbors have less influence)
                                distance = np.sqrt(dx**2 + dy**2 + dz**2)
                                weight = 1.0 / distance
                                neighbor_colors.append(voxel_colors[neighbor_idx])
                                neighbor_weights.append(weight)
                        except ValueError:
                            # this position doesn't have a voxel
                            pass

        # if we found neighbors with visibility information, use weighted average
        if neighbor_colors:
            total_weight = sum(neighbor_weights)
            weighted_color = np.zeros(3, dtype=np.float32)

            for color, weight in zip(neighbor_colors, neighbor_weights):
                weighted_color += np.array(color) * weight

            # update the color
            voxel_colors[idx] = weighted_color / total_weight

    # add all voxels to the data and colors lists
    for i, (x, y, z) in enumerate(voxel_positions):
        data.append(
            [
                x * block_size - width / 2,
                -(z * block_size - depth / 2),
                y * block_size,
            ]
        )
        colors.append(voxel_colors[i])

    if visualize_mesh:
        reconstructor.generate_mesh()

    return data, colors


def load_camera_parameters(camera_data_path):
    try:
        tree = ET.parse(camera_data_path)
        root = tree.getroot()

        camera_matrix = np.fromstring(
            root.find("CameraMatrix/data").text.replace("\n", " "), sep=" "
        ).reshape(3, 3)
        dist_coeffs = np.fromstring(
            root.find("DistortionCoeffs/data").text.replace("\n", " "), sep=" "
        )
        rvec = np.fromstring(
            root.find("RotationVector/data").text.replace("\n", " "), sep=" "
        )
        tvec = np.fromstring(
            root.find("TranslationVector/data").text.replace("\n", " "), sep=" "
        )
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
        camera_matrix, dist_coeffs, rvec, tvec = load_camera_parameters(
            camera_data_path
        )
        R, _ = cv2.Rodrigues(rvec)
        position = -np.matrix(R).T * np.matrix(tvec).T * scale
        # OpenGL uses Y-up, but the camera uses Z-up, so we need to flip the Y-axis
        cam_positions.append([position[0], -position[2], position[1]])

    cam_colors = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]
    return cam_positions, cam_colors


def get_cam_rotation_matrices():
    cam_rotations = []
    for i in range(1, 5):
        camera_data_path = f"./data/cam{i}/config.xml"
        camera_matrix, dist_coeffs, rvec, tvec = load_camera_parameters(
            camera_data_path
        )

        rotation_matrix, _ = cv2.Rodrigues(rvec)
        cam_rotation = np.identity(4)

        for row in range(3):
            for col in range(3):
                cam_rotation[row][col] = rotation_matrix[row][col]

        cam_rotation = cam_rotation.T

        flattened_matrix = cam_rotation.flatten()
        # rotate the camera 90 degrees around the Z-axis for correct viewing direction
        cam_rotations.append(
            glm.mat4(*flattened_matrix) * glm.rotate(np.pi / 2, glm.vec3(0, 0, 1))
        )
    return cam_rotations
