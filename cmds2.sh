for (( i=1; i<=30; i+=5 )); do checkpoint_path="/root/3DTrans/output/cfgs/DA/nusc_custom/voxelrcnn/voxel_rcnn_feat_3_vehi/default/ckpt/checkpoint_epoch_${i}.pth"; echo "Running checkpoint: $checkpoint_path"; python test.py --cfg_file ./cfgs/DA/nusc_custom/voxelrcnn/voxel_rcnn_feat_3_vehi.yaml --ckpt $checkpoint_path --batch_size 1; done

 bash scripts/ADA/dist_train_active_source.sh 2 --cfg_file ./cfgs/ADA/nuscenes-custom/voxel_rcnn/active_source.yaml --pretrained_model /root/3DTrans/output/cfgs/DA/nusc_custom/voxelrcnn/voxel_rcnn_feat_3_vehi/default/ckpt/checkpoint_epoch_16.pth

 bash scripts/ADA/dist_train_active.sh 2 --cfg_file ./cfgs/ADA/nuscenes-custom/voxel_rcnn/active_source.yaml --pretrained_model /root/3DTrans/output/cfgs/DA/nusc_custom/voxelrcnn/voxel_rcnn_feat_3_vehi/default/ckpt/checkpoint_epoch_16.pth


 bash scripts/ADA/dist_train_active.sh 2 --cfg_file ./cfgs/ADA/nuscenes-custom/voxel_rcnn/active_source.yaml --pretrained_model /root/3DTrans/output/cfgs/DA/nusc_custom/voxelrcnn/voxel_rcnn_feat_3_vehi/default/ckpt/checkpoint_epoch_16.pth

for (( i=1; i<=15; i+=1 )); do checkpoint_path="/root/3DTrans/output/cfgs/ADA/nuscenes-custom/voxel_rcnn/active_source/default/ckpt/checkpoint_epoch_${i}.pth"; echo "Running checkpoint: $checkpoint_path"; python test.py --cfg_file ./cfgs/ADA/nuscenes-custom/voxel_rcnn/active_source.yaml --ckpt $checkpoint_path --batch_size 1; done


for (( i=1; i<=30; i+=1 )); do checkpoint_path="/root/3DTrans/output/cfgs/DA/waymo_custom/voxel_rcnn_sn_custom/default/ckpt/checkpoint_epoch_${i}.pth"; echo "Running checkpoint: $checkpoint_path"; python test.py --cfg_file ./cfgs/DA/waymo_custom/voxel_rcnn_sn_custom.yaml --ckpt $checkpoint_path --batch_size 1; done


bash scripts/ADA/dist_train_active.sh 2 --cfg_file ./cfgs/ADA/nuscenes-custom/voxel_rcnn/active_dual_target_05.yaml --pretrained_model /root/3DTrans/output/cfgs/ADA/nuscenes-custom/voxel_rcnn/active_source/default/ckpt/checkpoint_epoch_1.pth


bash scripts/ADA/dist_train_active.sh 2 --cfg_file ./cfgs/ADA/nuscenes-custom/voxel_rcnn/active_dual_target_05.yaml --pretrained_model /root/3DTrans/output/cfgs/ADA/nuscenes-custom/voxel_rcnn/active_source/default/ckpt/checkpoint_epoch_1.pth --batch_size 6


n=0; while [[ $n -lt 15 ]]; do python test.py --cfg_file ./cfgs/ADA/waymo-custom/voxelrcnn/active_dual_target_05.yaml --ckpt /root/3DTrans/output/cfgs/ADA/waymo-custom/voxelrcnn/active_dual_target_05/default/ckpt/checkpoint_epoch_${n}.pth --batch_size 1 ; n=$((n+1)); done



n=0; while [[ $n -lt 10 ]]; do bash scripts/ADA/dist_train_active.sh 2 --cfg_file ./cfgs/ADA/nuscenes-custom/voxel_rcnn/active_dual_target_05.yaml --pretrained_model /root/3DTrans/output/cfgs/ADA/nuscenes-custom/voxel_rcnn/active_source/default/ckpt/checkpoint_epoch_1.pth --batch_size 6 ; n=$((n+1)); done

 bash scripts/ADA/dist_train_active_source.sh 2 --cfg_file ./cfgs/ADA/nuscenes-awsim/voxelrcnn/active_source.yaml --pretrained_model /root/3DTrans/output/cfgs/DA/nusc_custom/voxelrcnn/voxel_rcnn_feat_3_vehi/default/ckpt/checkpoint_epoch_16.pth --batch_size 6

n=1; while [[ $n -lt 30 ]]; do python test.py --cfg_file ./cfgs/DA/nusc_awsim/voxelrcnn/voxel_rcnn_feat_3_vehi.yaml --ckpt /root/3DTrans/output/cfgs/DA/nusc_awsim/voxelrcnn/voxel_rcnn_feat_3_vehi/default/ckpt/checkpoint_epoch_${n}.pth --batch_size 1 ; n=$((n+1)); done

n=1; while [[ $n -lt 15 ]]; do python test.py --cfg_file ./cfgs/ADA/nuscenes-awsim/voxelrcnn/active_source.yaml --ckpt /root/3DTrans/output/cfgs/ADA/nuscenes-awsim/voxelrcnn/active_source/v2/ckpt/checkpoint_epoch_${n}.pth --batch_size 1 ; n=$((n+1)); done

bash scripts/ADA/dist_train_active.sh 2 --cfg_file ./cfgs/ADA/nuscenes-awsim/voxelrcnn/active_dual_target_05.yaml --pretrained_model /root/3DTrans/output/cfgs/ADA/nuscenes-awsim/voxelrcnn/active_source/default/ckpt/checkpoint_epoch_1.pth --batch_size 4 --extra_tag v2

n=1; while [[ $n -lt 15 ]]; do bash scripts/ADA/dist_train_active.sh 2 --cfg_file ./cfgs/ADA/nuscenes-awsim/voxelrcnn/active_dual_target_05.yaml --pretrained_model /root/3DTrans/output/cfgs/ADA/nuscenes-awsim/voxelrcnn/active_source/default/ckpt/checkpoint_epoch_1.pth --batch_size 4 --extra_tag v2 ; n=$((n+1)); done


n=1; while [[ $n -lt 15 ]]; do python test.py --cfg_file ./cfgs/ADA/waymo-custom/active_TQS_1gpu.yaml --ckpt //root/3DTrans/output/cfgs/ADA/waymo-custom/active_TQS_1gpu/default/ckpt/checkpoint_epoch_${n}.pth --batch_size 1 ; n=$((n+1)); done



n=1; while [[ $n -lt 15 ]]; do python test.py --cfg_file ./cfgs/ADA/waymo-custom/voxelrcnn/active_CLUE.yaml --ckpt /root/3DTrans/output/cfgs/ADA/waymo-custom/voxelrcnn/active_CLUE/default/ckpt/checkpoint_epoch_${n}.pth --batch_size 1 ; n=$((n+1)); done


n=1; while [[ $n -lt 15 ]]; do bash scripts/ADA/dist_train_active_CLUE.sh 2 --cfg_file ./cfgs/ADA/waymo-awsim/active_CLUE.yaml --pretrained_model /root/3DTrans/output/cfgs/ADA/waymo-custom/pvrcnn_old_anchor/default/ckpt/checkpoint_epoch_15.pth --batch_size 1 ; n=$((n+1)); done

n=1; while [[ $n -lt 15 ]]; do python test.py --cfg_file ./cfgs/ADA/waymo-awsim/voxelrcnn/active_CLUE.yaml --ckpt /root/3DTrans/output/cfgs/ADA/waymo-awsim/voxelrcnn/active_CLUE/default/ckpt/checkpoint_epoch_${n}.pth --batch_size 1 ; n=$((n+1)); done
