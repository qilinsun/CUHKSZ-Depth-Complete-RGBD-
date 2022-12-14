import numpy as np
import time
import os
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import *
# from datasets.nyu import NYU
from datasets.mp import MP
from trainer_base import Trainer
import MinkowskiEngine as ME
from models.encoder3d import Encoder3D
from models.encoder2d import Encoder2D
from models.unet3d import UNet3D
from datasets.mp import customed_collate_fn
from tensorboardX import SummaryWriter
from PIL import Image

# Reference : https://github.com/nianticlabs/monodepth2

class Mp_trainer(Trainer):
    def __init__(self, options):
        super(Mp_trainer, self).__init__(options)
        
    def set_init(self):

        self.models = {}
        self.parameters_to_train = []
        self.is_eval = self.opt.is_eval
        self.dataset_name = 'matterport'
        self.data_path = self.opt.data_path
    
        # Networks
        self.models["enc2d"]  = Encoder2D(in_ch=4, output_dim=16)  
        self.models["enc3d"]  = Encoder3D(1, 16, D= 3, planes=(32, 48, 64)) 
        self.models["unet3d"] = UNet3D(32, self.opt.up_scale**2, f_maps=[32, 48, 64, 80], mode="nearest")
        
        if self.opt.load_model:
            self.load_model()
        
        for m in self.models:
            self.models[m].to(self.device)
            self.parameters_to_train += list(self.models[m].parameters())
            params = sum([np.prod(p.size()) for p in self.models[m].parameters()])
            print("# param of {}: {}".format(m,params))

        self.z_step = self.opt.max/(self.opt.res-1)
        
        self.silog_criterion = silog_loss(variance_focus=self.opt.variance_focus)
        self.gradient_loss = GradientLoss()
        # Train Mode
        if not self.is_eval:
            # Optimizer
            self.learning_rate = 0.0001
            self.scheduler_step_size = 10
            self.model_optimizer = torch.optim.Adam(self.parameters_to_train, self.learning_rate)
            self.model_lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self.model_optimizer, self.scheduler_step_size, 0.5)
            
    def process_batch(self, inputs, is_val = False):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            if key == 'scene_name':
                continue
            inputs[key] = ipt.to(self.device)
        
        outputs = {}
        losses = {}
        
        rgb = inputs["color"]
        dep = inputs["depth"]
        mask = inputs["mask"]
        gt = inputs["render_depth"]
        # gt_mask = inputs["gt_mask"]
        gt_mask = gt>0
        if self.opt.time:
            torch.cuda.synchronize()
            before_op_time = time.time()
        ##############################################################
        ## [step 1] RGB-D Feature Volume Construction
        in_2d = torch.cat([rgb, dep],1)
        in_3d = self.depth2MDP(dep)
        feat2d = self.models["enc2d"](in_2d)
        feat3d = self.models["enc3d"](in_3d)
        rgbd_feat_vol = self.fusion(feat3d, feat2d)

        ## [step 2] Cost Volume Prediction
        cost_vol = self.models["unet3d"](rgbd_feat_vol)
        
        ## [step 3] Depth Regression
        pred = self.upsampling(cost_vol, res = self.opt.res, up_scale=self.opt.up_scale) * self.z_step

        ###############################################################
        if self.opt.time:
            torch.cuda.synchronize()
            outputs["time"] = time.time() - before_op_time
            outputs["mem"] = torch.cuda.memory_allocated(self.device) / 1024 / 1024
            print(outputs["time"], outputs["mem"])
            
        outputs["depth"] = pred
        # losses["loss"] = L1_mask(gt, pred, gt_mask)
        # losses["loss"] = self.silog_criterion.forward(pred, gt, gt_mask.to(torch.bool))
        l1_loss = L1_mask(gt, pred, gt_mask)
        gradient_loss = self.gradient_loss(pred, gt)
        
        losses["loss"] = l1_loss + 0.7 * gradient_loss / (gradient_loss / l1_loss).detach()
        return outputs, losses

    def depth2MDP(self, dep):
        # Depth to sparse tensor in MDP (multiple-depth-plane)        
        idx = torch.round(dep / self.z_step).type(torch.int64)
        # Those depths that exceed to maximum depth plane are set to the last plane
        idx[idx>(self.opt.res-1)] = self.opt.res - 1
        # Those depths that smaller than the first plane are set to the first plane
        idx[idx<0] = 0
        # Get each point's "depth" at depth plane
        inv_dep = (idx * self.z_step)
        # Calculate the difference between real depth and depth plane, then normalize it
        res_map = (dep - inv_dep) / self.z_step

        B, C, H, W = dep.size()
        ones = (idx !=0).float()
        grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
        grid_ = torch.stack((grid_y, grid_x), 2).to(dep.device)
        # grid_ = self.grid.clone().detach()
        grid_ = grid_.unsqueeze(0).repeat((B,1,1,1))
        points_yx = grid_.reshape(-1,2)
        point_z = idx.reshape(-1, 1)
        m = (idx != 0).reshape(-1)
        points3d = torch.cat([point_z, points_yx], dim=1)[m]
        split_list = torch.sum(ones, dim=[1,2,3], dtype=torch.int).tolist()
        coords = points3d.split(split_list)
        # feat = torch.ones_like(points3d)[:,0].reshape(-1,1)       ## if occ to feat
        feat = res_map
        feat = feat.permute(0,2,3,1).reshape(-1, feat.size(1))[m]   ## if res to feat
        
        # Convert to a sparse tensor
        in_field = ME.TensorField(
            features = feat, 
            coordinates=ME.utils.batched_coordinates(coords, dtype=torch.float32),
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=dep.device,
        )
        return in_field.sparse() 

    def fusion(self, sout, feat2d):
        # sparse tensor to dense tensor
        B0,C0,H0,W0 = feat2d.size()
        dense_output_, min_coord, tensor_stride = sout.dense(min_coordinate=torch.IntTensor([0, 0, 0]))
        dense_output = dense_output_[:, :, :self.opt.res, :H0, :W0]
        B,C,D,H,W = dense_output.size() 
        feat3d_ = torch.zeros((B0, C0, self.opt.res, H0, W0), device = feat2d.device)
        feat3d_[:B,:,:D,:H,:W] += dense_output
        
        # construct type C feat vol
        mask = (torch.sum((feat3d_ != 0), dim=1, keepdim=True)!= 0).float()
        mask_ = mask + (1 - torch.sum(mask, dim=2,keepdim=True).repeat(1,1,mask.size(2),1,1))
        feat2d_ = feat2d.unsqueeze(2).repeat(1,1,self.opt.res,1,1) * mask_ 
        return torch.cat([feat2d_, feat3d_],dim = 1)
    
    def upsampling(self, cost, res = 64, up_scale = None):
        # if up_scale is None not apply per-plane pixel shuffle
        if not up_scale == None:
            b, c, d, h, w = cost.size()
            cost = cost.transpose(1,2).reshape(b, -1, h, w)
            cost = F.pixel_shuffle(cost, up_scale)
        else:
            cost = cost.squeeze(1)
        prop = F.softmax(cost, dim = 1)
        pred =  disparity_regression(prop, res)
        return pred

    def set_dataset(self):
        self.dataset = MP
        if not self.is_eval:
            train_data_path = os.path.join(self.data_path, 'train')    
            train_dataset = self.dataset(self.dataset_name, train_data_path, train = True)
            self.train_loader = DataLoader(
                train_dataset, self.opt.batch_size, True,
                num_workers=self.opt.num_workers, pin_memory=False, drop_last=False,
                collate_fn = customed_collate_fn(self.dataset_name))
            
        test_data_path = os.path.join(self.data_path, 'test')    
        test_dataset = self.dataset(self.dataset_name, test_data_path, train = False)
        self.val_loader = DataLoader(
            test_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=False, drop_last=False,
            collate_fn = customed_collate_fn(self.dataset_name))
        self.val_iter = iter(self.val_loader)
        if not self.is_eval:
            print('There are {} training items'.format(len(train_dataset)))
        print('There are {} testing items'.format(len(test_dataset)))
        
    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        if not self.is_eval:
            writer = self.writers[mode]
            for l, v in losses.items():
                writer.add_scalar("{}".format(l), v, self.step)
            
        
        dir = "{}/{}/{}".format(self.opt.gallary_path, mode, self.step)
        if not os.path.exists(dir):
            os.makedirs(dir)

        for j in range(min(len(inputs['depth']), self.opt.batch_size)):  # write a maxmimum of four images
            GT_dep = inputs["render_depth"][j].clone().detach()
            pred_dep = outputs["depth"][j].clone().detach()
            raw_dep = inputs["depth"][j].clone().detach()
            
            GT_dep = GT_dep.to(torch.device('cpu'))
            pred_dep = pred_dep.to(torch.device('cpu'))
            raw_dep = raw_dep.to(torch.device('cpu'))
            
            GT_dep = GT_dep.squeeze()
            pred_dep = pred_dep.squeeze()
            raw_dep = raw_dep.squeeze()
            
            GT_dep = GT_dep.mul_(4000).type(torch.int32).numpy()
            pred_dep = pred_dep.mul_(4000).type(torch.int32).numpy()
            raw_dep = raw_dep.mul_(4000).type(torch.int32).numpy()
            
            gt_im = Image.fromarray(GT_dep)
            pred_im = Image.fromarray(pred_dep)
            raw_im = Image.fromarray(raw_dep)
            gt_filename = os.path.join(dir, "{}_{}_{}.png".format("GT_depth", inputs["scene_name"][j], j))
            pred_dep_filename = os.path.join(dir, "{}_{}_{}.png".format("pred_dep", inputs["scene_name"][j], j))
            raw_dep_filename = os.path.join(dir, "{}_{}_{}.png".format("raw_dep", inputs["scene_name"][j], j))
            
            gt_im.save(gt_filename)
            pred_im.save(pred_dep_filename)
            raw_im.save(raw_dep_filename)
    
    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk"""
        
        assert os.path.isdir(self.opt.weight_path), \
            "Cannot find folder {}".format(self.opt.weight_path)
        print("loading model from folder {}".format(self.opt.weight_path))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.weight_path, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        # optimizer_load_path = os.path.join(self.opt.weight_path, "adam.pth")
        # if os.path.isfile(optimizer_load_path):
        #     print("Loading Adam weights")
        #     optimizer_dict = torch.load(optimizer_load_path)
        #     self.model_optimizer.load_state_dict(optimizer_dict)
        # else:
        #     print("Cannot find Adam weights so Adam is randomly initialized")
    
    
    def evaluate(self, is_offline= False):
        """Run the entire training pipeline
        # """
        if is_offline:
            path = '{}/*'.format(self.opt.weight_path)
            models = sorted(glob.glob(path))
            for m in models:
                print('Load weights :{}'.format(m))
                name = (m.split('/')[-1]).split('.')[0]
                if name not in ['adam', 'sche'] :
                    self.models[name].load_state_dict(torch.load(m, map_location=self.device), strict=False)

        print("evaling")
        self.set_eval()
        result = torch.empty((0,8),device = self.device)
        times = []
        mems = []
        with torch.no_grad(): 
            for batch_idx, inputs in enumerate(self.val_loader):
                outputs, losses = self.process_batch(inputs, is_val = True)
                batch = outputs["depth"].size(0)
                for i in range(batch):
                    pred = outputs["depth"][i].unsqueeze(0)
                    gt = inputs["render_depth"][i].unsqueeze(0)
                    mini_result = evaluate(gt, pred, losses, is_test = True)
                    result = torch.vstack([result, mini_result])
                    self.log('eval', inputs, outputs, losses)
                    if self.opt.time:
                        times.append(outputs["time"])
                        mems.append(outputs["mem"])

        fin = torch.mean(result, dim = 0)

        if self.opt.time:
            print('Avg time:', np.mean(times), 'Avg Mem:', np.mean(mems))
        print_string = ">>> [Test] RMSE {:.4f} | REL {:.4f} | d1: {:.4f} | d2: {:.4f} | d3: {:.4f} "
        print(print_string.format(fin[0], fin[4], fin[5], fin[6], fin[7]))
        exit()
    
    
from options import Options

options = Options()
opts = options.parse()

if __name__ == "__main__":
    print('Testing mode')
    
    now = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    model_name = f'crnn_{now}'  
    log_path = os.path.join("./runs", model_name)

    writer = SummaryWriter(os.path.join(log_path, 'testing'))
    dataset = MP
    dataset_name = 'matterport'
    train_data_path = os.path.join(opts.data_path, 'train')    
    train_dataset = dataset(dataset_name, train_data_path, train = True)
    train_loader = DataLoader(
        train_dataset, opts.batch_size, True,
        num_workers=opts.num_workers, pin_memory=False, drop_last=False,
        collate_fn = customed_collate_fn(dataset_name))
    
    
    step = 0
    for batch_idx, inputs in enumerate(train_loader):
         for j in range(min(4, opts.batch_size)):  # write a maxmimum of four images
             
            GT_dep = inputs["render_depth"][j].clone().detach()
            
            GT_dep = GT_dep.to(torch.device('cpu'))
            
            GT_dep = GT_dep.squeeze()
            
            GT_dep = GT_dep.mul_(4000).type(torch.int32).numpy()
            
            im = Image.fromarray(GT_dep)
            im.save("./gallary/GT_depth/{}_{}.png".format(inputs["scene_name"][j], j))
            
            # writer.add_image(
            #     "raw_dep_{}_{}".format(inputs["scene_name"][j], j),
            #     dep, step
            # )
            
            writer.add_image(
                "GT_dep_{}_{}".format(inputs["scene_name"][j], j),
                inputs["render_depth"][j], step
            )
            
            step += 1

        

