import xml.etree.ElementTree as ET
import cv2
import numpy as np
import pickle
import os
from tqdm import tqdm
from background_subtraction import BackgroundSubtractor
import concurrent.futures  # For parallelization
import glm

class VoxelReconstruction:
    def __init__(self, base_camera_data_path, width, height, depth, block_size, frame_dims, num_cameras=4, num_threads=4, min_cameras_required=4):
        self.base_camera_data_path = base_camera_data_path #directory with all camera configurations
        self.width = width
        self.height = height
        self.depth = depth
        self.block_size = block_size
        self.frame_dims = frame_dims
        self.num_cameras = num_cameras #number of cameras
        self.min_cameras_required = min_cameras_required  # Minimum cameras to agree

        self.cameras = self.load_all_camera_data() # Load camera parameters from individual files
        self.voxel_space = np.zeros((self.width, self.height, self.depth), dtype=np.uint8)  # Binary occupancy grid
        self.voxel_colors = np.zeros((self.width, self.height, self.depth, 3), dtype=np.uint8) #Store color of each voxel
        self.voxel_counts = np.zeros((self.width, self.height, self.depth), dtype=np.uint8) #store the count of each voxel, instead of clipping

        self.lookup_table = self.load_lookup_table()  # Try loading first
        if self.lookup_table is None:
            self.num_threads = num_threads
            self.lookup_table = self.create_lookup_table()  # Replaces project_space
            self.save_lookup_table() #save newly created table
        self.previous_masks = {} #store previous masks for efficiency
        self.last_voxel_space = None #store the result of last reconstruction.

    def load_all_camera_data(self):
        """Loads camera parameters from individual XML files."""
        cameras = []
        for i in range(1, self.num_cameras + 1):
            camera_data_path = os.path.join(self.base_camera_data_path, f"cam{i}", "config.xml") #path for each file
            camera = self.parse_camera_data(camera_data_path, i) #pass camera id
            cameras.append(camera)
        return cameras

    def parse_camera_data(self, path, camera_id):
        """Parses the XML file containing camera data."""
        try:
            root = ET.parse(path).getroot()
            camera = {}
            camera['id'] = camera_id #assign camera id
            camera['CameraMatrix'] = np.fromstring(root.find('CameraMatrix/data').text.replace('\n', ' '), sep=' ').reshape(3,3)
            camera['DistortionCoefficients'] = np.fromstring(root.find('DistortionCoeffs/data').text.replace('\n', ' '), sep=' ')
            camera['RotationVectors'] = np.fromstring(root.find('RotationVector/data').text.replace('\n', ' '), sep=' ')
            camera['TranslationVectors'] = np.fromstring(root.find('TranslationVector/data').text.replace('\n', ' '), sep=' ')
            return camera
        except FileNotFoundError:
            print(f"Error: Camera configuration file not found at {path}")
            return None
        except Exception as e:
            print(f"Error parsing XML: {e}")
            return None

    def _process_voxel(self, camera, x, y, z):
        """Helper function to process a single voxel."""
        voxel = np.array([
            x * self.block_size - self.width / 2 * self.block_size,
            y * self.block_size,
            z * self.block_size - self.depth / 2 * self.block_size,
        ], dtype=np.float32)

        projected, _ = cv2.projectPoints(voxel,
                                       camera['RotationVectors'],
                                       camera['TranslationVectors'],
                                       camera['CameraMatrix'],
                                       camera['DistortionCoefficients'])
        projected = np.int32(projected).squeeze()

        if self.is_point_in_bounds(projected):
            return (tuple(projected), (x, y, z))  # Return pixel and voxel
        return None

    def create_lookup_table(self):
        """
        Creates and returns the lookup table for voxel projections in parallel.
        """
        lookup_table = {}  # Dictionary to store lookup tables for each camera

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            for camera in self.cameras:
                camera_id = camera['id']
                lookup_table[camera_id] = {}
                # Prepare a list of tasks for the executor
                tasks = []
                for x in range(self.width):
                    for y in range(self.height):
                        for z in range(self.depth):
                            tasks.append((camera, x, y, z))

                # Submit tasks to the executor and process results as they come in
                for result in tqdm(executor.map(lambda task: self._process_voxel(task[0], task[1], task[2], task[3]), tasks), total=len(tasks), desc=f"Creating Lookup Table for Camera {camera_id}"):
                    if result:
                        pixel, voxel = result
                        if pixel not in lookup_table[camera_id]:
                            lookup_table[camera_id][pixel] = []
                        lookup_table[camera_id][pixel].append(voxel)

        return lookup_table

    def is_point_in_bounds(self, point):
        """
        Checks if a 2D point is within the image bounds.
        """
        return 0 <= point[0] < self.frame_dims[0] and 0 <= point[1] < self.frame_dims[1]

    def reconstruct_from_frames(self):
        """Reconstruct the voxel model from video frames."""
        voxel_changes = {}
        color_changes = {}
        for camera in self.cameras:
            if camera is None:  # Handle potential loading errors.
                print("Skipping camera due to loading error during reconstruction.")
                continue

            camera_id = camera['id']
            background_model = BackgroundSubtractor(f'./data/cam{camera_id}/background.avi', f'./data/cam{camera_id}/video.avi')
            mask, frame = background_model.isolate_foreground()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #set to RGB

            # Optimize: Check for silhouette changes using XOR with previous mask
            if camera_id in self.previous_masks:
                delta_mask = cv2.bitwise_xor(mask, self.previous_masks[camera_id])
                changed_pixels = np.transpose(np.where(delta_mask == 255)) #get indices of changes

            else:
                changed_pixels = None #no previous mask, process all pixels
            self.previous_masks[camera_id] = mask # Store the current mask

            self.back_project_silhouette(mask, frame, camera_id, changed_pixels, voxel_changes, color_changes) #also pass frame for color

        self.update_voxel_space(voxel_changes, color_changes) #update voxel space and color.
        return self.voxel_space

    def back_project_silhouette(self, binary_mask, frame, camera_id, changed_pixels, voxel_changes, color_changes):
        """Back-projects the silhouette from a camera into the voxel space."""

        if changed_pixels is not None: #optimized case: only process changes
            for pixel in changed_pixels:
                x, y = pixel[1], pixel[0] # openCV matrix indexing is swapped
                pixel = (x,y) #pixel value becomes the new key
                if pixel in self.lookup_table[camera_id]:
                    voxels = self.lookup_table[camera_id][pixel]

                    # If pixel is foreground, mark voxels as potentially occupied
                    if binary_mask[y, x] == 255:
                        color = frame[y, x] #grab color

                        for vx, vy, vz in voxels:
                            if (vx, vy, vz) not in voxel_changes:
                                voxel_changes[(vx, vy, vz)] = 0
                            voxel_changes[(vx, vy, vz)] += 1  # Increment voxel count

                            if (vx, vy, vz) not in color_changes:
                                color_changes[(vx, vy, vz)] = np.array([0, 0, 0], dtype=np.float32) #using float32 for precision
                            color_changes[(vx, vy, vz)] += color.astype(np.float32)  # Add the color for averaging (cast to float for precision)

        else: #no mask, process all pixels
            for y in range(binary_mask.shape[0]):
                for x in range(binary_mask.shape[1]):
                    pixel = (x, y)
                    if pixel in self.lookup_table[camera_id]:
                        voxels = self.lookup_table[camera_id][pixel]
                        # If pixel is foreground, mark voxels as potentially occupied
                        if binary_mask[y, x] == 255:
                            color = frame[y, x]
                            for vx, vy, vz in voxels:
                                if (vx, vy, vz) not in voxel_changes:
                                    voxel_changes[(vx, vy, vz)] = 0
                                voxel_changes[(vx, vy, vz)] += 1  # Increment voxel count

                                if (vx, vy, vz) not in color_changes:
                                    color_changes[(vx, vy, vz)] = np.array([0, 0, 0], dtype=np.float32)
                                color_changes[(vx, vy, vz)] += color.astype(np.float32)

    def update_voxel_space(self, voxel_changes, color_changes):
        """Updates the voxel space based on voxel changes."""
        for (vx, vy, vz), change in voxel_changes.items():
            self.voxel_counts[vx, vy, vz] = self.voxel_counts[vx, vy, vz] + change #increment voxel count
            self.voxel_space[vx, vy, vz] = 1 if self.voxel_counts[vx,vy,vz] >= self.min_cameras_required else 0 #update based on min_cameras

        for (vx, vy, vz), color_sum in color_changes.items():
            if self.voxel_space[vx, vy, vz] == 1: #only update if voxel is occupied
                self.voxel_colors[vx, vy, vz] = (color_sum / self.voxel_counts[vx,vy,vz]).astype(np.uint8) #average the colors

    def save_lookup_table(self, filename="lookup_table.pkl"):
        """Saves the lookup table to a file."""
        with open(filename, 'wb') as f:
            pickle.dump(self.lookup_table, f)
        print(f"Lookup table saved to {filename}")

    def load_lookup_table(self, filename="lookup_table.pkl"):
        """Loads the lookup table from a file."""
        try:
            with open(filename, 'rb') as f:
                lookup_table = pickle.load(f)
            print(f"Lookup table loaded from {filename}")
            return lookup_table
        except FileNotFoundError:
            print("Lookup table file not found. Creating a new one.")
            return None

#Example usage
if __name__ == '__main__':
    #Example Usage
    base_camera_data_path = "./data"  # Directory containing cam1, cam2, etc.
    width, height, depth = 64, 64, 64  # Example voxel space dimensions
    block_size = 15.0
    frame_dims = (644, 486)
    num_cameras = 4
    num_threads = 8
    min_cameras_required = 4 #min number of cameras that have to see voxel to be considered foreground

    reconstruction = VoxelReconstruction(base_camera_data_path, width, height, depth, block_size, frame_dims, num_cameras, num_threads, min_cameras_required)
    voxel_model = reconstruction.reconstruct_from_frames() #reconstruct from video
    print("Voxel reconstruction complete.")

    # Now you can use voxel_model for rendering or further processing.
    #To access the voxel colors, use reconstruction.voxel_colors