import xml.etree.ElementTree as ET
import cv2
import numpy as np
import pickle
import os
from tqdm import tqdm
from background_subtraction import BackgroundSubtractor

class VoxelReconstructor:
    def __init__(self, width, height, depth, frame_dim) -> None:
        self.frame_dim = frame_dim
        self.lookup_table = np.empty((4, frame_dim[0], frame_dim[1]), dtype=object)

        self.cameras = self._load_cameras()
        self.width, self.height, self.depth = width, height, depth
        self.scale = 25
        self.project_space()
        self.voxel_space = np.zeros((self.width, self.height, self.depth, 4), dtype=np.uint8)

        for camera in self.cameras:
            camera_id = camera['id']
            background_model = BackgroundSubtractor(f'./data/cam{camera_id}/video.avi', f'./data/cam{camera_id}/background.avi')
            mask, frame = background_model.isolate_foreground()
            voxel_space = self.back_project_silhouette(mask, frame, camera_id)
            self.voxel_space += voxel_space

    def back_project_silhouette(self, binary_mask, image, camera_id):
        voxel_space = np.zeros((self.width, self.height, self.depth, 4), dtype=np.uint8)
        contour_2d = []
        for x in range(binary_mask.shape[0]):
            for y in range(binary_mask.shape[1]):
                if binary_mask[x][y] > 0:
                    contour_2d.append((y, x))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        contour_2d = np.array(contour_2d).squeeze()
        contour_voxels = []
        contour_colors = []
        for points in tqdm(contour_2d):
            try:
                point_voxels = self.lookup_table[camera_id - 1, int(points[0]), int(points[1])]
            except:
                continue
            if point_voxels is None:
                continue
            
            # Store each voxel in the set along with the corresponding color
            for voxel in point_voxels:
                contour_voxels.append(voxel)
                contour_colors.append(image[int(points[1]), int(points[0])])  # Note: x,y order for image indexing
        
        # Process each voxel and its color
        for voxel, color in zip(contour_voxels, contour_colors):
            # Unpack the voxel coordinates from the tuple
            x, y, z = voxel
            voxel_space[x, y, z, 0] += 1
            voxel_space[x, y, z, 1] = color[0]
            voxel_space[x, y, z, 2] = color[1]
            voxel_space[x, y, z, 3] = color[2]
            
        print(f'Voxel space for camera {camera_id} has {np.sum(voxel_space[:,:,:,0])} voxels out of {self.width * self.height * self.depth}')
        return voxel_space

    def project_space(self):
        if os.path.exists('lookup_table.pkl'):
            with open('lookup_table.pkl', 'rb') as file:
                self.lookup_table = pickle.load(file)
            return
                                  
        for camera in self.cameras:
            for x in tqdm(range(self.width)):
                for y in range(self.height):
                    for z in range(self.depth):
                        voxel = np.array([
                            x * self.scale - self.width / 2 * self.scale, 
                            y * self.scale, 
                            z * self.scale - self.depth / 2 * self.scale,
                                        ], dtype=np.float32)
                        
                        projected, _ = cv2.projectPoints(voxel, #np.dot(voxel, rotation_matrix), 
                                                       camera['RotationVectors'],
                                                       camera['TranslationVectors'], 
                                                       camera['CameraMatrix'], 
                                                       camera['DistortionCoefficients'])
                        projected = np.int32(projected).squeeze()
                        # print(f'Camera {camera["id"]} projected {voxel} to {projected}')
                        if not self.point_in_bounds(projected):
                            continue
                        if self.lookup_table[camera['id'] - 1, projected[0], projected[1]] is None:
                            self.lookup_table[camera['id'] - 1, projected[0], projected[1]] = set()
                        self.lookup_table[camera['id'] - 1, projected[0], projected[1]].add((x, y, z))

        with open('lookup_table.pkl', 'wb') as file:
            pickle.dump(self.lookup_table, file)
        return
    
    def point_in_bounds(self, point):
        return not (point[0] < 0 or point[0] >= self.frame_dim[0] or point[1] < 0 or point[1] >= self.frame_dim[1])
    
    def _load_cameras(self):
        """Load camera parameters for all cameras."""
        cameras = []
        for i in range(1, 5):  # Assuming 4 cameras
            camera_data_path = f"./data/cam{i}/config.xml"
            camera_matrix, dist_coeffs, rvec, tvec = self._load_camera_parameters(camera_data_path)
            cameras.append({
                'id': i,
                'CameraMatrix': camera_matrix,
                'DistortionCoefficients': dist_coeffs,
                'RotationVectors': rvec,
                'TranslationVectors': tvec
            })
        return cameras
    
    def _load_camera_parameters(self, camera_data_path):
        """Load camera parameters from XML file."""
        try:
            tree = ET.parse(camera_data_path)
            root = tree.getroot()

            camera_matrix = np.fromstring(root.find('CameraMatrix/data').text.replace('\n', ' '), sep=' ').reshape(3, 3)
            dist_coeffs = np.fromstring(root.find('DistortionCoeffs/data').text.replace('\n', ' '), sep=' ')
            rvec = np.fromstring(root.find('RotationVector/data').text.replace('\n', ' '), sep=' ')
            tvec = np.fromstring(root.find('TranslationVector/data').text.replace('\n', ' '), sep=' ')
            return camera_matrix, dist_coeffs, rvec, tvec
        except Exception as e:
            print(f"Error loading camera parameters: {e}")
            return None, None, None, None