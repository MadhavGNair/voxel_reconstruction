import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import binary_erosion
from skimage import measure


class MarchingCubesMeshGenerator:
    def __init__(self, voxel_space, width, height, depth, block_size=1.0):
        self.voxel_space = voxel_space
        self.width = width
        self.height = height
        self.depth = depth
        self.block_size = block_size

        # extract occupancy data (first channel indicates how many cameras see this voxel)
        self.occupancy = voxel_space[:, :, :, 0].astype(np.float32)

        # extract color data (normalized RGB)
        self.colors = np.zeros((width, height, depth, 3), dtype=np.float32)
        for x in range(width):
            for y in range(height):
                for z in range(depth):
                    if self.occupancy[x, y, z] > 0:
                        # average the color values by the number of cameras that see this voxel
                        camera_count = self.occupancy[x, y, z]
                        self.colors[x, y, z, 0] = (
                            voxel_space[x, y, z, 1] / camera_count / 255.0
                        )  # R
                        self.colors[x, y, z, 1] = (
                            voxel_space[x, y, z, 2] / camera_count / 255.0
                        )  # G
                        self.colors[x, y, z, 2] = (
                            voxel_space[x, y, z, 3] / camera_count / 255.0
                        )  # B

    def generate_mesh(self, threshold=4, step_size=1):
        # apply threshold to occupancy data
        volume = self.occupancy.copy()

        # threshold the volume to create a binary volume
        # voxels seen by at least 'threshold' cameras are considered solid
        volume = (volume >= threshold).astype(np.float32)

        # apply marching cubes to get the mesh
        try:
            verts, faces, normals, values = measure.marching_cubes(
                volume[::step_size, ::step_size, ::step_size],
                level=0.5,  # threshold for isosurface
            )

            # scale vertices to match the original voxel space dimensions
            verts = verts * step_size

            # apply block size scaling
            verts = verts * self.block_size

            # center the model (optional)
            verts[:, 0] -= (self.width * self.block_size) / 2
            verts[:, 2] -= (self.depth * self.block_size) / 2

            return verts, faces, normals, values
        except Exception as e:
            print(f"Error generating mesh: {e}")
            return None, None, None, None

    def visualize_mesh(self, verts, faces, save_path=None):
        if verts is None or faces is None:
            print("No mesh to visualize")
            return

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")

        # create mesh collection
        mesh = Poly3DCollection(verts[faces])
        mesh.set_edgecolor("k")
        ax.add_collection3d(mesh)

        # set axis limits
        max_range = (
            np.array(
                [
                    verts[:, 0].max() - verts[:, 0].min(),
                    verts[:, 1].max() - verts[:, 1].min(),
                    verts[:, 2].max() - verts[:, 2].min(),
                ]
            ).max()
            / 2.0
        )

        mid_x = (verts[:, 0].max() + verts[:, 0].min()) / 2
        mid_y = (verts[:, 1].max() + verts[:, 1].min()) / 2
        mid_z = (verts[:, 2].max() + verts[:, 2].min()) / 2

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Visualization saved to {save_path}")

        plt.show()


def generate_mesh_from_voxels(
    voxel_space, width, height, depth, threshold=3.5, step_size=1, visualize=True
):
    # create mesh generator
    mesh_generator = MarchingCubesMeshGenerator(voxel_space, width, height, depth)

    # generate mesh
    verts, faces, normals, values = mesh_generator.generate_mesh(threshold, step_size)

    if verts is not None:
        if visualize:
            mesh_generator.visualize_mesh(verts, faces)

    return verts, faces, normals, values
