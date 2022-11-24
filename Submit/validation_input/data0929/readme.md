### Description

* folder

  * depth:

    *  深度图: 文件格式为`.png` ，分辨率$640 \times 480$, 数据格式为`uint16`, depth scale $1000$（即像素值对应单位为mm, 转化为m为单位需要除以1000） ;

    * 将深度图转化为以`m`为单位:

      ```python
      meter_image = depth_image / 1000.0
      ```
      
    
  * infrared

    * 红外图：`png`格式， 分辨率$640 \times 480$, 数据格式为`uint16`，整体偏暗，可以针对他做一个local tone mapping, or single image HDR进行亮度调整

  * rgb:

    * 对齐到深度图的rgb图， 对于深度图缺失的区域无法进行几何映射，故为缺失: `.jpg` 文件，分辨率$640 \times 480$

      

  * origin_RGB:
    * 原始RGB图: `.png`格式，分辨率$1920 \times 1080$

* Intrinsics_extrinsics: intrinsic for RGB camera and ToF camera; extrinsic from RGB to ToF

  ```json
  width 1920, height 1080
  K: 
     1124.708130, 0.000000, 935.169434,
     0.000000, 1125.579224, 556.538696,
     0.000000, 0.000000, 1.000000
  distortion:
     model: BROWN
     distortion k1: 0.069894, k2: -0.083615, k3: 0.018774, p1: -0.001987, p2: 0.000004
  R|T 
  0.999955, 0.002266, 0.009252, 0.045097
  -0.002245, 0.999995, -0.002235, 0.000998
  -0.009257, 0.002215, 0.999955, 0.000775
  0.000000, 0.000000, 0.000000, 1.000000
  width 640, height 480
  K: 
     531.781433, 0.000000, 324.895294,
     0.000000, 531.548645, 229.041000,
     0.000000, 0.000000, 1.000000
  distortion:
     model: BROWN
     distortion k1: -0.227852, k2: 0.041467, k3: 0.093066, p1: 0.000451, p2: 0.000581
  R|T 
  1.000000, 0.000000, 0.000000, 0.000000
  0.000000, 1.000000, 0.000000, 0.000000
  0.000000, 0.000000, 1.000000, 0.000000
  0.000000, 0.000000, 0.000000, 1.000000
  width 640, height 480
  K: 
     531.781433, 0.000000, 324.895294,
     0.000000, 531.548645, 229.041000,
     0.000000, 0.000000, 1.000000
  distortion:
     model: BROWN
     distortion k1: -0.227852, k2: 0.041467, k3: 0.093066, p1: 0.000451, p2: 0.000581
  R|T 
  1.000000, 0.000000, 0.000000, 0.000000
  0.000000, 1.000000, 0.000000, 0.000000
  0.000000, 0.000000, 1.000000, 0.000000
  0.000000, 0.000000, 0.000000, 1.000000
  ```

  

### Align RGB to ToF






































