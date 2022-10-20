# 2022/10/19

## depth_RGB alignment

$$R*P_{rgb} + T = P_{dep3d}$$
$$P_{rgb} = R^{-1}*P_{dep3d} - R^{-1}*T$$

### Result
Result of the aligned depth map are in folder "/EMDC-Pytorch/Submit/calidation_input/data0929/align_depth_rgb" and "/aligned_depth"
![avatar](/EMDC-PyTorch/Submit/validation_input/data0929/align_depth_rgb/frame-000000.color.png)

### Problem
1. There are grids in the image. Probably because RGB image is 1920\*1080 while depth image is only 640\*480. Reasons not clear so far.
2. Large depth missing region in the surrounding area.

## Test in EMDC model

### Modification
1. As the model uses MobileNetv2, There are some downsampling and upsampling block. If the size of the input image is not sepecificlly chosen, odd dimension of the feature map may appear during the downsampling process. <br> For example, the input image is 1920\*1080, after several downsamplings, the size will be 960\*540 -> 480\*270 -> 240\*135. After the that, the dimension will round to a integer number, 120\*68.<br> As a result, when we do unsampling, the feature map of 120\*68 will be unsampled to 240\*136. This is a problem because when doing contacatenation, the dimension won't align.<br> In order to solve this problem, I simply zero_padding the inital 1920\*1080 image into a 1920\*1088 image. **But I'm not sure whether doing this will affect the performence of the model.**

### Result
Result are in folder  EMDC-PyTorch/Submit/results/EMDC-validation/visual/data0929/aligned_depth/.

[link](EMDC-PyTorch/Submit/results/EMDC-validation/visual/data0929/aligned_depth/frame-000001.depth_jet.png)

1. It will show the grid-like feature in the input aligned depth map.
2. The predictions in the corresponding depth-missing area is not good.

[link](EMDC-PyTorch/Submit/results/EMDC-validation/visual/data0929/aligned_depth/frame-000002.depth_jet.png)

### Solutions

For the second problem, cropping the image when doing depth-rgb alignment may work.<br>
Have not found out the solution to the first problem.