import os
import sys
sys.path.append("/root/3DTrans/")
datasets =["kitti_models", "custom", "awsim"]
models =["centerpoint","pv_rcnn", "voxel_rcnn_3_class"]

cfgs =  []
ckpts = []

for model in models:
    for dataset in datasets:
        cfg = "/root/3DTrans/tools/cfgs/"+dataset+"/" +model+".yaml"
        cfgs.append(cfg)
        ckpt = "/root/3DTrans/output/root/3DTrans/tools/cfgs/"+dataset+"/"+model+"/default/ckpt/checkpoint_epoch_80.pth"
        if os.path.isfile(ckpt):
            ckpts.append(ckpt)
        else:
            ckpt = "/root/3DTrans/output/root/3DTrans/tools/cfgs/"+dataset+"/"+model+"/default/ckpt/checkpoint_epoch_30.pth"
            if os.path.isfile(ckpt):
                ckpts.append(ckpt)
            else:
                print(ckpt," can not found!!")
                exit()
                      
for cfg,ckpt in zip(cfgs,ckpts):
    os.system("python test.py --cfg_file "+cfg +" --ckpt "+ckpt +" --batch_size 1" )