project_name='resnet50_training_method_sif_duke_epoch50_stepsize10_lr_0.1'
mkdir logs
mkdir "logs/$project_name"
CUDA_VISIBLE_DEVICES=0 python main.py \
--project_name $project_name \
--dataset 'duke-list' \
--b 2 \
--lr 0.1 \
--a resnet50 \
--training_method sif \
--data_dir '/home/longwei.wl/reid/open_reid_weilong/datasets' \
--features 512 \
--ncls 703 \
--step_size 10 \
--epochs 30 
