import sys
sys.path.append('./')

import yaml
import numpy as np
import cv2
from src.sfs import VoxelSpace
from src.utils import visualize_pcd

YAML_FILE_PATH = "config/params.yaml"

def get_params():
    with open(YAML_FILE_PATH, encoding='utf-8')as f:
        params = yaml.safe_load(f)

    return params

def main():
    params = get_params()
    voxel_space = VoxelSpace(params["x_range"], params["y_range"], params["z_range"], params["voxel_size"], np.array(params["K"]),params["focal_length"],params["table_height"])

    for i in range(params["num_images"]):
        image = cv2.imread(f"images/view{i+1}.png",0)
        extrinsic = np.array(params[f"extrinsic{i+1}"])
        voxel_space.sfs(image,extrinsic)

        pcd, num_points = voxel_space.pointcloud
        print(f"num_points : {num_points}")
        visualize_pcd(pcd)


if __name__ == "__main__":
    main()
    


    
    