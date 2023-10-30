import numpy as np
import open3d as o3d

class VoxelSpace:
    def __init__(self, x_range, y_range, z_range, voxel_size,K):
        # number of voxel in each axis
        self.voxel_number = [np.abs(x_range[1] - x_range[0]) / voxel_size[0], np.abs(y_range[1] - y_range[0]) / voxel_size[1], np.abs(z_range[1] - z_range[0]) / voxel_size[2]]
        
        # total number of voxels
        self.total_number = np.prod(self.voxel_number).astype(int)
        
        # (total_number, 4)
        # The first three values are the x-y-z-coordinates of the voxel, the fourth value is the occupancy
        self.voxel = np.ones((self.total_number, 4))
        

        l = 0
        for x in range(int(self.voxel_number[0])):
            for y in range(int(self.voxel_number[1])):
                for z in range(int(self.voxel_number[2])):
                    self.voxel[l] = [x * voxel_size[0], y * voxel_size[1], z * voxel_size[2], 0] 
                    l += 1

        
        self.points3D = np.copy(self.voxel).T
        self.points3D[3,:] = 1
        
        # camera intrinsic
        self.K = K

        # number of images used for SfS
        self.num_image = 0
        

    def sfs(self, image, extrinsic):
        self.num_image += 1

        height, width, image, silhouette = self.preprocess_image(image)

        #perspective projection matrix
        p_matrix = self.calc_p_matrix(extrinsic)

        
        # projection to the image plane (points2D = (u,v,1) * 41^3)
        points2D = np.matmul(p_matrix, self.points3D)
        points2D = np.floor(points2D / points2D[2, :]).astype(np.int32) # 3行目を1に揃える


        ind1 = np.where((points2D[0, :] < 0) | (points2D[0, :] >= width)) # check for u value bigger than width
        ind2 = np.where((points2D[1, :] < 0) | (points2D[1, :] >= height))  # check for v value bigger than width
        ind = ind1 + ind2

        # accumulate the value of each voxel in the current image
        tmp = silhouette[points2D.T[:,1], points2D.T[:,0]]
        tmp[ind[1]] = False
        self.voxel[:,3] += tmp

        self.visualize_pcd(self.num_image)


    def calc_p_matrix(self,extrinsic):
        return np.matmul(self.K,extrinsic)
    
    def preprocess_image(self,image):
        height, width = np.shape(image)
        image[np.where(image != 0)] = 1
        silhouette = image > 0
        return height, width, image, silhouette
    
    def visualize_pcd(self,num_image):
        ind = self.voxel[:,3] >= num_image

        # get pointcloud from voxel
        self.pcd = self.voxel[ind,:]
        # get xyz coordinates of pointcloud
        self.pcd = self.pcd[:,0:3]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.pcd)

        o3d.io.write_point_cloud("pointcloud.ply", pcd)
        # o3d.visualization.draw_geometries([pcd])

        viewer = o3d.visualization.Visualizer()
        viewer.create_window()
        viewer.add_geometry(pcd)
        opt = viewer.get_render_option()
        opt.show_coordinate_frame = True
        viewer.run()
        viewer.destroy_window()

