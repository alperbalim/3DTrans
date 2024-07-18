import pickle as pkl

infos = pkl.load(open("/root/3DTrans/data/custom_kitti/kitti_infos_train.pkl","rb"))

for info in infos:
    print("ID:  ",info['point_cloud']["lidar_idx"], "  Label:  ",info['annos']["name"][0], "        Points: ",info['annos']["num_points_in_gt"][0])