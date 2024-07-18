import pickle as pkl


path = "/root/3DTrans/data/custom_kitti/kitti_infos_train.pkl"
path="/root/3DTrans/data/custom_kitti2/custom_infos_test.pkl"
with open(path, 'rb') as f:
    infos = pkl.load(f)
len(infos)
print(info[0])