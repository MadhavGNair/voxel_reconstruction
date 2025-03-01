import xml.etree.ElementTree as ET
import cv2
import numpy as np
import pickle
import os
import time
import concurrent.futures
import multiprocessing as mp
from tqdm import tqdm
from background_subtraction import BackgroundSubtractor

class VoxelReconstructor:
    def __init__(self, width, height, depth, frame_dim) -> None:
        self.frame_dim = frame_dim
        self.lookup_table = np.empty((4, frame_dim[0], frame_dim[1]), dtype=object)
        # Store camera positions for depth reasoning
        self.camera_positions = []

        self.num_cameras = 4
        self.cameras = self._load_cameras()
        self.width, self.height, self.depth = width, height, depth
        self.scale = 25
        self.lookup_table_path = f"./lookup_table_{self.width}x{self.height}x{self.depth}_{self.scale}.pkl"
        self._load_lookup_table()
        self.voxel_space = np.zeros((self.width, self.height, self.depth, 4), dtype=np.uint8)
        # Store depth information for each voxel from each camera
        self.depth_map = np.zeros((self.num_cameras, self.width, self.height, self.depth), dtype=np.float32)
        # Store visibility information for each voxel from each camera
        self.visibility_map = np.zeros((self.num_cameras, self.width, self.height, self.depth), dtype=np.bool_)
        # Store color information for each voxel from each camera
        self.color_map = np.zeros((self.num_cameras, self.width, self.height, self.depth, 3), dtype=np.uint8)

    def _load_cameras(self):
        """Load camera parameters for all cameras."""
        cameras = []
        for i in range(1, self.num_cameras + 1):  # Assuming 4 cameras
            camera_data_path = f"./data/cam{i}/config.xml"
            camera_matrix, dist_coeffs, rvec, tvec = self._load_camera_parameters(camera_data_path)
            # Calculate camera position in world coordinates
            R, _ = cv2.Rodrigues(rvec)
            position = -np.matrix(R).T * np.matrix(tvec).T
            self.camera_positions.append(position.flatten())
            
            cameras.append({
                'id': i,
                'CameraMatrix': camera_matrix,
                'DistortionCoefficients': dist_coeffs,
                'RotationVectors': rvec,
                'TranslationVectors': tvec,
                'Position': position.flatten()
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

    def _process_camera_process_pool(self, camera_data):
        """
        Process a single camera's voxels for the lookup table.
        This function is designed to be used with ProcessPoolExecutor.
        """
        camera_id, rotation_vectors, translation_vectors, camera_matrix, distortion_coeffs = camera_data
        
        # Create a local lookup table for this camera
        local_lookup = {}
        
        # Process all voxels for this camera
        for x in tqdm(range(self.width), desc=f"Processing voxels for camera {camera_id}", leave=False):
            for y in range(self.height):
                for z in range(self.depth):
                    voxel = np.array([
                        x * self.scale - self.width / 2 * self.scale, 
                        y * self.scale, 
                        z * self.scale - self.depth / 2 * self.scale,
                    ], dtype=np.float32)
                    
                    projected, _ = cv2.projectPoints(
                        voxel,
                        rotation_vectors,
                        translation_vectors, 
                        camera_matrix, 
                        distortion_coeffs
                    )
                    
                    projected = np.int32(projected).squeeze()
                    
                    # Check if point is in bounds
                    if (projected[0] < 0 or projected[0] >= self.frame_dim[0] or 
                        projected[1] < 0 or projected[1] >= self.frame_dim[1]):
                        continue
                    
                    # Calculate distance from camera to voxel for depth ordering
                    camera_pos = self.cameras[camera_id-1]['Position']
                    distance = np.linalg.norm(voxel - camera_pos)
                    
                    # Store in local dictionary with depth information
                    key = (projected[0], projected[1])
                    if key not in local_lookup:
                        local_lookup[key] = []
                    local_lookup[key].append((x, y, z, distance))
        
        return camera_id, local_lookup

    def _load_lookup_table(self):
        """
        Creates the lookup table using ProcessPoolExecutor for parallelization.
        This approach uses separate processes to avoid GIL limitations.
        """
        print(f"Creating lookup table with ProcessPoolExecutor for dimensions {self.width}x{self.height}x{self.depth} with scale {self.scale}")
        
        # Initialize lookup table
        self.lookup_table = np.empty((self.num_cameras, self.frame_dim[0], self.frame_dim[1]), dtype=object)
        
        # Measure execution time
        start_time = time.time()
        
        # Prepare camera data for processing
        camera_data_list = []
        for camera in self.cameras:
            camera_data = (
                camera['id'],
                camera['RotationVectors'],
                camera['TranslationVectors'],
                camera['CameraMatrix'],
                camera['DistortionCoefficients']
            )
            camera_data_list.append(camera_data)
        
        # Use ProcessPoolExecutor to process cameras in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(4, mp.cpu_count())) as executor:
            futures = []
            for camera_data in camera_data_list:
                futures.append(executor.submit(self._process_camera_process_pool, camera_data))
            
            # Process results as they complete
            for future in tqdm(concurrent.futures.as_completed(futures), 
                              total=len(futures),
                              desc="Processing cameras with ProcessPoolExecutor"):
                try:
                    camera_id, local_lookup = future.result()
                    
                    # Transfer local lookup to the main lookup table
                    for (px, py), voxel_list in local_lookup.items():
                        # Sort voxels by distance from camera (closest first)
                        voxel_list.sort(key=lambda v: v[3])
                        
                        # Store sorted voxels in lookup table
                        self.lookup_table[camera_id - 1, px, py] = voxel_list
                        
                except Exception as exc:
                    print(f'Camera processing exception: {exc}')
        
        # Calculate and print execution time
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"ProcessPoolExecutor lookup table creation completed in {execution_time:.2f} seconds")
        
        # Save the lookup table
        print(f"Saving lookup table to {self.lookup_table_path}")
        with open(self.lookup_table_path, 'wb') as file:
            pickle.dump(self.lookup_table, file)
        return

    def _process_pixel_batch(self, camera_id, rgb_frame, batch_pixels):
        """Process a batch of pixels in parallel with depth reasoning"""
        batch_results = []
        
        for point in batch_pixels:
            try:
                point_voxels = self.lookup_table[camera_id - 1, int(point[0]), int(point[1])]
                if point_voxels is not None:
                    color = rgb_frame[int(point[1]), int(point[0])]
                    
                    # Get the closest voxel to the camera (first in sorted list)
                    if len(point_voxels) > 0:
                        closest_voxel = point_voxels[0]
                        x, y, z, distance = closest_voxel
                        batch_results.append(((x, y, z), color, distance))
                        
                        # Mark other voxels along this ray as occluded
                        for voxel_info in point_voxels[1:]:
                            x, y, z, _ = voxel_info
                            batch_results.append(((x, y, z), color, float('inf')))  # Infinite distance means occluded
            except:
                continue
                
        return batch_results

    def _update_voxel_batch(self, voxel_batch, temp_voxel_space):
        """Process a batch of voxels for final voxel space construction with visibility information"""
        local_voxel_space = np.zeros((self.width, self.height, self.depth, 4), dtype=np.uint8)
        count = 0
        
        for voxel, color, distance in voxel_batch:
            x, y, z = voxel
            if temp_voxel_space[x, y, z] > 0:
                local_voxel_space[x, y, z, 0] = 1  # Mark as occupied
                local_voxel_space[x, y, z, 1:] = color  # Set color
                count += 1
                
        return local_voxel_space, count

    def reconstruct_voxels(self, masks_and_frames, camera_id):
        # Initialize voxel spaces
        temp_voxel_space = np.zeros((self.width, self.height, self.depth), dtype=np.uint8)
        voxel_space = np.zeros((self.width, self.height, self.depth, 4), dtype=np.uint8)
        
        if not masks_and_frames:
            return voxel_space
            
        # Process first frame to initialize potential voxels
        first_mask, first_frame = masks_and_frames[0]
        rgb_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        
        # Create a dictionary to store potential voxels and their colors
        voxel_dict = {}
        # Store visibility information
        voxel_visibility = {}
        
        # Process first frame completely to initialize all potential voxels
        foreground_pixels = []
        for x in range(first_mask.shape[0]):
            for y in range(first_mask.shape[1]):
                if first_mask[x][y] > 0:
                    foreground_pixels.append((y, x))
        
        foreground_pixels = np.array(foreground_pixels).squeeze()
        if foreground_pixels.size == 0:
            return voxel_space
            
        # Handle case where only one pixel is in foreground_pixels
        if len(foreground_pixels.shape) == 1:
            foreground_pixels = np.array([foreground_pixels])
            
        print(f"Processing {len(foreground_pixels)} pixels from first frame")
        
        # Use thread-based concurrency for sharing lookup table (which is already in memory)
        # This avoids pickle issues with self.lookup_table in multiprocessing
        voxel_list = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Calculate adaptive batch size using square root rule
            total_pixels = len(foreground_pixels)
            cpu_count = os.cpu_count() or 4
            base_size = int(np.sqrt(total_pixels))
            adaptive_batch_size = max(100, base_size // cpu_count)
            
            print(f"Using adaptive batch size of {adaptive_batch_size} for {total_pixels} pixels on {cpu_count} cores")
            
            # Create batches with adaptive size
            batches = [foreground_pixels[i:i + adaptive_batch_size] for i in range(0, total_pixels, adaptive_batch_size)]
            
            # Submit all batches to executor - passing camera_id and rgb_frame once
            futures = []
            for batch in batches:
                futures.append(executor.submit(self._process_pixel_batch, camera_id, rgb_frame, batch))
            
            # Collect results as they complete
            for future in tqdm(concurrent.futures.as_completed(futures), 
                              total=len(futures),
                              desc=f"Initializing voxels for camera {camera_id}"):
                try:
                    results = future.result()
                    voxel_list.extend(results)
                except Exception as exc:
                    print(f'Batch processing exception: {exc}')
        
        # Process results with visibility information
        for voxel, color, distance in voxel_list:
            # Store color information
            voxel_dict[voxel] = color
            temp_voxel_space[voxel] = 1
            
            # Store visibility information (distance < inf means visible)
            is_visible = distance < float('inf')
            voxel_visibility[voxel] = is_visible
            
            # Update color map and visibility map
            if is_visible:
                self.color_map[camera_id-1, voxel[0], voxel[1], voxel[2]] = color
                self.visibility_map[camera_id-1, voxel[0], voxel[1], voxel[2]] = True
                self.depth_map[camera_id-1, voxel[0], voxel[1], voxel[2]] = distance
        
        # Keep track of previous mask for XOR operation
        prev_mask = first_mask.copy()
        
        # For each subsequent frame - process sequentially as each depends on previous
        for i, (mask, frame) in enumerate(tqdm(masks_and_frames[1:], desc=f"Processing frames for camera {camera_id}")):
            # Use XOR to find pixels that changed between frames
            diff_mask = cv2.bitwise_xor(mask, prev_mask)
            
            # Update previous mask for next iteration
            prev_mask = mask.copy()
            
            # Create a new temporary voxel space for this frame
            current_frame_voxels = temp_voxel_space.copy()
            
            # Process only pixels that changed (using XOR)
            changed_pixels = []
            for x in range(diff_mask.shape[0]):
                for y in range(diff_mask.shape[1]):
                    if diff_mask[x][y] > 0:  # This pixel changed
                        changed_pixels.append((y, x))
            
            changed_pixels = np.array(changed_pixels).squeeze()
            if changed_pixels.size > 0:
                # Handle case where only one pixel changed
                if len(changed_pixels.shape) == 1:
                    changed_pixels = np.array([changed_pixels])
                                    
                # For each changed pixel
                for points in changed_pixels:
                    try:
                        point_voxels = self.lookup_table[camera_id - 1, int(points[0]), int(points[1])]
                    except:
                        continue
                    if point_voxels is None:
                        continue
            
                    # Check if this pixel is now foreground or background
                    is_foreground = mask[int(points[1]), int(points[0])] > 0
                    
                    for voxel_info in point_voxels:
                        x, y, z, _ = voxel_info
                        voxel = (x, y, z)
                        if not is_foreground:
                            # If pixel became background, remove corresponding voxels
                            current_frame_voxels[voxel] = 0
                            # Also update visibility
                            if voxel in voxel_visibility:
                                voxel_visibility[voxel] = False
                                self.visibility_map[camera_id-1, x, y, z] = False
            
            # Update the temporary voxel space with the current frame's voxels
            temp_voxel_space = current_frame_voxels
            
            # If no voxels remain consistent, break early
            if np.sum(temp_voxel_space) == 0:
                print(f"No consistent voxels remain after frame {i+1}")
                break
        
        # Transfer consistent voxels to final voxel space with color information
        consistent_voxel_count = 0
        voxel_items = list(voxel_dict.items())
        
        # Using ThreadPoolExecutor for final voxel space construction
        # This phase has few dependencies and operates on pre-existing data
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Calculate adaptive batch size using square root rule
            total_voxels = len(voxel_items)
            cpu_count = os.cpu_count() or 4
            base_size = int(np.sqrt(total_voxels))
            adaptive_batch_size = max(100, base_size // cpu_count)
            
            print(f"Using adaptive batch size of {adaptive_batch_size} for {total_voxels} voxels on {cpu_count} cores")
            
            # Create batches with adaptive size
            batches = [voxel_items[i:i + adaptive_batch_size] for i in range(0, total_voxels, adaptive_batch_size)]
            
            # Submit batches for processing - now with simplified parameters
            futures = []
            for batch in batches:
                # Convert batch to include visibility information
                batch_with_visibility = [(voxel, color, 0) for voxel, color in batch]
                futures.append(executor.submit(self._update_voxel_batch, batch_with_visibility, temp_voxel_space))
            
            # Process results as they complete
            for future in tqdm(concurrent.futures.as_completed(futures), 
                              total=len(futures),
                              desc=f"Finalizing voxel space for camera {camera_id}"):
                try:
                    local_space, count = future.result()
                    # Combine local space into the final voxel space
                    # For each non-zero voxel in local_space, copy its values to voxel_space
                    non_zero_indices = np.where(local_space[..., 0] > 0)
                    for idx in range(len(non_zero_indices[0])):
                        x, y, z = non_zero_indices[0][idx], non_zero_indices[1][idx], non_zero_indices[2][idx]
                        voxel_space[x, y, z, 0] = 1  # Mark as seen by this camera
                        
                        # Only set color if this voxel is visible from this camera
                        if self.visibility_map[camera_id-1, x, y, z]:
                            voxel_space[x, y, z, 1:] = local_space[x, y, z, 1:]
                    
                    consistent_voxel_count += count
                except Exception as exc:
                    print(f'Voxel space update exception: {exc}')
        
        print(f'Voxel space for camera {camera_id} has {consistent_voxel_count} consistent voxels out of {self.width * self.height * self.depth} possible voxels')
        return voxel_space

    def reconstruct(self):
        """
        Performs the voxel reconstruction process using all camera angles.
        """
        self.voxel_space = np.zeros((self.width, self.height, self.depth, 4), dtype=np.uint8)
        
        # Initialize visibility and color maps
        self.visibility_map = np.zeros((self.num_cameras, self.width, self.height, self.depth), dtype=np.bool_)
        self.color_map = np.zeros((self.num_cameras, self.width, self.height, self.depth, 3), dtype=np.uint8)
        self.depth_map = np.full((self.num_cameras, self.width, self.height, self.depth), float('inf'), dtype=np.float32)
        
        # Process each camera
        for camera in self.cameras:
            camera_id = camera['id']
            background_model = BackgroundSubtractor(f'./data/cam{camera_id}/video.avi', f'./data/cam{camera_id}/background.avi')
            masks_and_frames = background_model.isolate_foreground()
            voxel_space = self.reconstruct_voxels(masks_and_frames, camera_id)
            
            # Add camera's contribution to voxel space
            for x in range(self.width):
                for y in range(self.height):
                    for z in range(self.depth):
                        if voxel_space[x, y, z, 0] > 0:
                            self.voxel_space[x, y, z, 0] += 1
                            
                            # Only update color if this camera has visibility
                            if self.visibility_map[camera_id-1, x, y, z]:
                                self.voxel_space[x, y, z, 1:] += voxel_space[x, y, z, 1:]
        
        return self.voxel_space


    