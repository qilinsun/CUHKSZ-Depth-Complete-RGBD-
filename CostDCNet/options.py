import os
import argparse

file_dir = os.path.dirname(__file__) 

class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="CostDCNet options")
        self.parser.add_argument("--num_epochs", type=int, help="total epochs", default=50)
        self.parser.add_argument("--batch_size", type=int, help="batch size", default=16)
        self.parser.add_argument("--num_workers", type=int, help="number of dataloader workers", default=8)
        self.parser.add_argument("--log_frequency", type=int, help="number of batches between each tensorboard log", default=250)
        self.parser.add_argument("--save_frequency", type=int, help="number of epochs between each save", default=2)

        self.parser.add_argument('--gpu', type=int, default = 0, help='gpu id')
        self.parser.add_argument("--data_path", type=str, default = 'data0929', help="path of dataset")
        self.parser.add_argument("--gallary_path", type=str, default = 'data0929/result', help="path of result")
        self.parser.add_argument('--weight_path', type=str, default = 'CostDCNet/runs/crnn_20221204_201943/models/weights_11', help='path of pretrained weights')
        self.parser.add_argument('--models_to_load', type=str, default = ["enc2d", "enc3d", "unet3d"], help='path of pretrained weights')
        self.parser.add_argument('--is_eval', type=bool, default = True, help='evaluation')
        self.parser.add_argument('--time', type=bool,default = False, help='sec')
        self.parser.add_argument('--load_model', type=bool, default = True, help='Load models weights')
        self.parser.add_argument('--width', type=int, default = 320, help='image width')
        self.parser.add_argument('--height', type=int, default = 256, help='image width')
        self.parser.add_argument('--max', type=float, default = 15.0, help='maximum depth value')
        self.parser.add_argument('--res', type=int, default = 16, help='number of depth plane')
        self.parser.add_argument('--up_scale', type=int, default = 4, help='scale factor of upsampling')

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
