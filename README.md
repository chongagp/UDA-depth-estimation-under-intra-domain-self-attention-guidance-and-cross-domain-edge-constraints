# UDA-depth-estimation-under-intra-domain-self-attention-guidance-and-cross-domain-edge-constraints
![Framework](https://github.com/chongagp/UDA-depth-estimation-under-intra-domain-self-attention-guidance-and-cross-domain-edge-constraints/blob/main/img/framework.jpg)
## Environment
1. Python 3.7
2. PyTorch 1.8.0
3. CUDA 10.0
4. Ubuntu 20.04

## Datasets
[KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php)

[vKITTI](https://europe.naverlabs.com/Research/Computer-Vision/Proxy-Virtual-Worlds/)

[Make3D](http://make3d.cs.cornell.edu/data.html)(FOR TEST)


## Training (NVIDIA GeForce RTX 3090, 16GB)

- Train F_t
```
python train.py --model ft --gpu_ids 0 --batchSize 8 --loadSize 256 1024 --g_tgt_premodel ./cyclegan/G_Tgt.pth
```

- Train F_s
```
python train.py --model fs --gpu_ids 0 --batchSize 8 --loadSize 256 1024 --g_src_premodel ./cyclegan/G_Src.pth
```

- Train TOTAL using the pretrained F_s, F_t and CycleGAN.
```
python train.py --freeze_bn --freeze_in --model Total --gpu_ids 0 --batchSize 3 --loadSize 192 640 --g_src_premodel ./cyclegan/G_Src.pth --g_tgt_premodel ./cyclegan/G_Tgt.pth --d_src_premodel ./cyclegan/D_Src.pth --d_tgt_premodel ./cyclegan/D_Tgt.pth --t_depth_premodel ./checkpoints/vkitti2kitti_ft_bn/**_net_G_Depth_T.pth --s_depth_premodel ./checkpoints/vkitti2kitti_fs_bn/**_net_G_Depth_S.pth 
```

## Test

Copy the provided models to ./checkpoints/vkitti2kitti_Total/, and rename the models 1_* (e.g., 1_net_D_Src.pth), and then
```
python test.py --test_datafile 'test.txt' --which_epoch 1 --model Total --gpu_ids 0 --batchSize 1 --loadSize 192 640
```

## Contact
Peng Guo: 230218956@seu.edu.cn
