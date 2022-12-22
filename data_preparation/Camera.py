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

        self.rgb_intrinsic = np.array([[1105.848267, 0.000000, 979.189026],
                                        [0.000000, 1105.758423, 533.063416],
                                        [0.000000, 0.000000, 1.000000]])

        self.depth_intrinsic = np.array([[530.325317, 0.000000, 321.097351],
                                        [0.000000, 530.411377, 246.448624],
                                        [0.000000, 0.000000, 1.000000]])

        self.rgb_distortion = np.array([0.042573, -0.135081, 0.117709, -0.000113, 0.000281])
        self.depth_distortion = np.array([-0.231988, 0.111533, -0.042759, 0.000236, -0.000089])

        self.rotation_matrix = np.array([[0.999985, -0.004662, -0.002922],
                                        [0.004663, 0.999989, 0.000428],
                                        [0.002920, -0.000442, 0.999996]])
                    
        self.translation_matrix = np.array([-0.043315, 0.002525, 0.006187], dtype=np.float64)
        
        self.extrinsic = np.array([[0.999985, -0.004662, -0.002922, -0.043315],
                                    [0.004663, 0.999989, 0.000428, 0.002525],
                                    [0.002920, -0.000442, 0.999996, 0.006187],
                                    [0.0, 0.0, 0.0, 1.0]])

        # self.doffs = 342.523