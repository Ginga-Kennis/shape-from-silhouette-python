import numpy as np
import open3d as o3d

from src.utils import visualize_pcd


class VoxelSpace:
    def __init__(self, x_range, y_range, z_range, voxel_size,K,focal_length):
        # number of voxel in each axis
        self.voxel_number = [np.abs(x_range[1] - x_range[0]) / voxel_size[0], np.abs(y_range[1] - y_range[0]) / voxel_size[1], np.abs(z_range[1] - z_range[0]) / voxel_size[2]]
        
        # total number of voxels
        self.total_number = np.prod(self.voxel_number).astype(int)
        
        # (total_number, 4)
        # The first three values are the x-y-z-coordinates of the voxel, the fourth value is the occupancy(0 or 1)
        self.voxel = np.ones((self.total_number, 4))

        # total number of images projected to each points
        self.num_projected = np.zeros((self.total_number, 1))

        l = 0
        for z in range(1,int(self.voxel_number[2])+1):
            for y in range(1,int(self.voxel_number[1])+1):
                for x in range(1,int(self.voxel_number[0])+1):
                    self.voxel[l] = [voxel_size[0] * (x - 0.5),voxel_size[1] * (y - 0.5),voxel_size[2] * (z - 0.5), 0] 
                    l += 1

        
        self.points3D_world = np.copy(self.voxel).T
        self.points3D_world[3,:] = 1
        
        # camera intrinsic
        self.K = K
        self.f = focal_length
        

    def sfs(self, image, extrinsic):
        height, width, image, silhouette = self.preprocess_image(image)

        #perspective projection matrix
        extrinsic = np.linalg.inv(extrinsic) 
        p_matrix = self.calc_p_matrix(extrinsic[0:3,:])

        # projection to the image plane (points2D = (u,v,1) * self.total_number)
        points2D = np.matmul(p_matrix, self.points3D_world)
        points2D = np.floor(points2D / points2D[2, :]).astype(np.int32)

        # check for points less than focal length
        points3D_camera = np.matmul(extrinsic,self.points3D_world)
        ind1 = np.where((points3D_camera[2,:] < self.f))

        # check for (u < 0, width < u) and (v < 0, height < v)
        ind2 = np.where(((points2D[0, :] < 0) | (points2D[0, :] >= width) | (points2D[1, :] < 0) | (points2D[1, :] >= height))) 
        points2D[:,ind2] = 0  # just for error handling

        # concat ind1 and ind2
        ind = np.unique(np.concatenate((ind1[0],ind2[0])))

    
        # 0 : outside image
        # 1 : inside image
        projected = np.ones((self.total_number, 1))
        projected[ind,0] = 0.0


        # 0 : inside silhouette
        # 1 : outside silhouette
        # 2 : outside image
        tmp = silhouette[points2D.T[:,1], points2D.T[:,0]].astype(int)
        tmp[ind] = 2

        
        for i in range(self.total_number):
            if self.voxel[i,3] == 0.0:
                # 0 → 1 (only when self.num_projected == 0)
                if tmp[i] == 1 and self.num_projected[i] == 0:
                    self.voxel[i,3] = 1.0
            else:
                # 1 → 0
                if tmp[i] == 0:
                    self.voxel[i,3] = 0.0
        self.remove_table()
        self.num_projected += projected

            

    def remove_table(self):
        self.voxel[np.where(self.voxel[:,2] < 0.05)[0],3] = 0

    def calc_p_matrix(self,extrinsic):
        return np.matmul(self.K,extrinsic)
    
    def preprocess_image(self,image):
        height, width = np.shape(image)
        image[np.where(image != 0)] = 1
        silhouette = image > 0
        return height, width, image, silhouette


    @property
    def pointcloud(self):
        ind = np.where(self.voxel[:,3] == 1.0)
        pcd = self.voxel[ind[0],0:3]
        num_points = np.shape(pcd)[0]
        return pcd, num_points
    
    @property
    def voxel_space(self):
        voxel_space = self.voxel[:,3]
        return voxel_space.reshape(int(self.voxel_number[0]),int(self.voxel_number[1]),int(self.voxel_number[2]))




        
    


    


