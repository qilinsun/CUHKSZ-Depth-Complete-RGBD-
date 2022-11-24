import numpy as np

class Camera:

    def __init__(self) -> None:
        ''' 
            Using MiddleBury Dataset here.
            No need to undistort as the dataset has done that.
            camera intrinsic matrices:
                cam0=[7190.247 0 1035.513; 0 7190.247 945.196; 0 0 1]
                cam1=[7190.247 0 1378.036; 0 7190.247 945.196; 0 0 1]
            doffs:    x-difference of principal points, doffs = cx1 - cx0 
            baseline: camera baseline in mm
        '''

        self.rgb_width, self.rgb_height = 1920, 1080
        self.depth_width, self.depth_height = 640, 480

        self.depth_intrinsic = np.array([[531.781433, 0.000000, 324.895294],
                                         [0.000000, 531.548645, 229.041000],
                                         [0.000000, 0.000000, 1.000000]])

        self.rgb_intrinsic = np.array([[1124.708130, 0.000000, 935.169434],
                                       [0.000000, 1125.579224, 556.538696],
                                       [0.000000, 0.000000, 1.000000]])

        self.depth_distortion = np.array([0.069894, -0.083615, 0.018774, -0.001987, 0.000004])
        self.rgb_distortion = np.array([-0.227852, 0.041467, 0.093066, 0.000451, 0.000581])

        self.rotation_matrix = np.array([[0.999955, 0.002266, 0.009252],
                                [-0.002245, 0.999995, -0.002235],
                                [-0.009257, 0.002215, 0.999955]])
                    
        self.translation_matrix = np.array([0.055097, 0.000998, 0.000775], dtype=np.float64)*1000

        # self.doffs = 342.523