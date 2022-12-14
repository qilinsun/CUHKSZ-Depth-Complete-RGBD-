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
        self.parser.add_argument("--save_frequency", type=int, help="number of epochs between each save", default=5)

        self.parser.add_argument('--gpu', type=int, default = 0, help='gpu id')
        self.parser.add_argument("--data_path", type=str, default = '/data/tong/matterport', help="path of dataset")
        self.parser.add_argument("--gallary_path", type=str, default = './gallary/eval_gallary/gallary_silog', help="path of result")
        self.parser.add_argument('--weight_path', type=str, default = 'CostDCNet/weights/silog_loss/', help='path of pretrained weights')
        self.parser.add_argument('--models_to_load', type=str, default = ["enc2d", "enc3d", "unet3d"], help='path of pretrained weights')
        self.parser.add_argument('--is_eval', type=bool, default = True, help='evaluation')
        self.parser.add_argument('--time', type=bool,default = False, help='sec')
        self.parser.add_argument('--load_model', type=bool, default = True, help='Load models weights')
        self.parser.add_argument('--width', type=int, default = 320, help='image width')
        self.parser.add_argument('--height', type=int, default = 256, help='image width')
        self.parser.add_argument('--max', type=float, default = 15.0, help='maximum depth value')
        self.parser.add_argument('--res', type=int, default = 16, help='number of depth plane')
        self.parser.add_argument('--up_scale', type=int, default = 4, help='scale factor of upsampling')
        
        self.parser.add_argument('--variance_focus', type=float, default = 0.85, help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error')


    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
