import sys
sys.path.append('../')
print(sys.path)

import argparse
import numpy as np
from src.sfs import VoxelSpace

def main(args):
    K = np.array(args.K)
    voxel = VoxelSpace(args.x_range,args.y_range,args.z_range,args.voxel_size,K)


    voxel.sfs(image,extrinsic)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--x_range", help="length of the x-axis in voxel space in meters. (example) [0.0, 0.3]", default = [0.0,0.3])
    parser.add_argument("--y_range", help="length of the y-axis in voxel space in meters. (example) [0.0, 0.3]", default = [0.0,0.3])
    parser.add_argument("--z_range", help="length of the z-axis in voxel space in meters. (example) [0.0, 0.3]", default = [0.0,0.3])
    parser.add_argument("--voxel_size", help="voxel size of each axis in meters. (example) [0.0075, 0.0075, 0.0075]" ,default = [0.0075,0.0075,0.0075])
    parser.add_argument("--K", help="Camera intrinsic parameters.3×3 matrix", default = [[891.318115234375, 0.0, 628.9073486328125],
                                                                              [0.0, 891.318115234375, 362.3601989746094],
                                                                              [0.0, 0.0, 1.0]])
    args = parser.parse_args()
    main(args)
    


    
    