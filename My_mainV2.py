#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
import os
import gc
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from My_modelV2 import DCP
from My_utilV2 import transform_point_cloud, npmat2euler, ICPIter
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from My_ReadCADV2 import ModelNet40H5, ModelNet40PCD

from scipy.spatial.transform import Rotation

import open3d as o3d
import copy


def printT(data):
    print("================================")
    print(data)
    print("--------------------------------")
    print('Shape: ', data.shape)
    print('dType: ', data.dtype)
    print('Device: ', data.device)
    print("================================\n")

class Counter():
    def __init__(self, cnt):
        self.target = cnt
        self.count = 0
    def Count(self):
        self.count += 1
        if (self.count >= self.target):
            return True
        else:
            return False
    def Reset(self):
        self.count = 0

class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()
            

def test_one_epoch(args, net, test_loader):
    net.eval()

    total_loss = 0
    num_examples = 0
    rotations_ab = []
    translations_ab = []
    rotations_ab_pred = []
    translations_ab_pred = []

    rotations_ba = []
    translations_ba = []
    rotations_ba_pred = []
    translations_ba_pred = []

    eulers_ab = []
    eulers_ba = []
    
    counter = Counter(50)
    ANGLE_LIST = []
    TRANS_LIST = []
    ANGLE_ROT_LIST = []
    TRANS_LIST = []
    
    for src, target, rotation_ab, translation_ab, rotation_ba, translation_ba, euler_ab, euler_ba in tqdm(test_loader):
        
        if args.cudaF:
            src = src.cuda()
            target = target.cuda()
            rotation_ab = rotation_ab.cuda()
            translation_ab = translation_ab.cuda()
            rotation_ba = rotation_ba.cuda()
            translation_ba = translation_ba.cuda()

        batch_size = src.size(0)
        num_examples += batch_size
        rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred = net(src, target)

        ## save rotation and translation
        rotations_ab.append(rotation_ab.detach().cpu().numpy())
        translations_ab.append(translation_ab.detach().cpu().numpy())
        rotations_ab_pred.append(rotation_ab_pred.detach().cpu().numpy())
        translations_ab_pred.append(translation_ab_pred.detach().cpu().numpy())
        eulers_ab.append(euler_ab.numpy())
        ##
        
        rotations_ba.append(rotation_ba.detach().cpu().numpy())
        translations_ba.append(translation_ba.detach().cpu().numpy())
        rotations_ba_pred.append(rotation_ba_pred.detach().cpu().numpy())
        translations_ba_pred.append(translation_ba_pred.detach().cpu().numpy())
        eulers_ba.append(euler_ba.numpy())


        transformed_src = transform_point_cloud(src, rotation_ab_pred, translation_ab_pred)

        transformed_target = transform_point_cloud(target, rotation_ba_pred, translation_ba_pred)

        ###########################
        if (args.cudaF):
            identity = torch.eye(3).cuda().unsqueeze(0).repeat(batch_size, 1, 1)
        else:
            identity = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1)
        loss = F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity, reduction = 'sum') \
               + F.mse_loss(translation_ab_pred, translation_ab, reduction = 'sum')

        total_loss += loss.item()

        if (args.cudaF):
            src = src.cpu()
            target = target.cpu()
            transformed_src = transformed_src.cpu()
            transformed_target = transformed_target.cpu()
        
        src = np.squeeze(src.numpy().T, axis = 2)
        target = np.squeeze(target.numpy().T, axis = 2)
        transformed_src = np.squeeze(transformed_src.detach().numpy().T, axis = 2)
        transformed_target = np.squeeze(transformed_target.detach().numpy().T, axis = 2)

        srcPC = o3d.geometry.PointCloud()
        srcPC.points = o3d.utility.Vector3dVector(src)
        srcPC.paint_uniform_color([1, 0.4, 0.4])
        targetPC = o3d.geometry.PointCloud()
        targetPC.points = o3d.utility.Vector3dVector(target)
        targetPC.paint_uniform_color([0, 0, 1])
        
        
        srcPC.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=10))
        srcFPFH = o3d.pipelines.registration.compute_fpfh_feature(
            srcPC, o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=10))
        
        targetPC.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=10))
        tarFPFH = o3d.pipelines.registration.compute_fpfh_feature(
            targetPC, o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=10))
        
        DCP_ROT = np.squeeze(rotation_ab_pred.detach().cpu().numpy(), axis = 0)
        DCP_TRANS = translation_ab_pred.detach().cpu().numpy().T
        DCP_TRANSFORM = np.block([[DCP_ROT, DCP_TRANS], [np.eye(4)[-1]]])
        
        #========Visualizer Setting========
        # viewer = o3d.visualization.Visualizer()
        # viewer.create_window('Viewer', 600, 600)
        
        # ctr = o3d.visualization.ViewControl()
        # ctr.set_zoom(2)
        # ctr.set_front([1, 1, 1])
        # ctr.set_lookat([0, 0, 0])
        # ctr.set_up([0, 1, 0])
        
        # viewer.add_geometry(srcPC)
        # viewer.add_geometry(targetPC)
        
        #========FGR Setting========
        # FGR = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        #     copy.deepcopy(srcPC), copy.deepcopy(targetPC), srcFPFH, tarFPFH)
        
        # FGR_ICP = o3d.pipelines.registration.registration_icp(
        #     copy.deepcopy(srcPC), copy.deepcopy(targetPC), 1, FGR.transformation, 
        #     o3d.pipelines.registration.TransformationEstimationPointToPlane())
        
        # FGR_ICP_PC = copy.deepcopy(srcPC)
        # FGR_ICP_PC.transform(FGR_ICP.transformation)
        # FGR_ICP_PC.paint_uniform_color([1, 0, 1])
        
        #========DCP Setting========
        # DCP_ICP_PC = copy.deepcopy(srcPC)     
        # DCP_ICP_PC.paint_uniform_color([0, 1, 0])
        # viewer.add_geometry(DCP_ICP_PC)
        # DCP_ICP_PC.transform(DCP_TRANSFORM)
        # iterSize = 50
        # for i in range(iterSize):
        #     DCP_ICP_TRANSFORM = ICPIter(DCP_ICP_PC, targetPC, np.identity(4), iterSize = 1)
        #     DCP_ICP_PC.transform(DCP_ICP_TRANSFORM.transformation)
        #     viewer.update_geometry(DCP_ICP_PC)
        #     viewer.poll_events()
        #     viewer.update_renderer()
        
        DCP_ICP_PC = copy.deepcopy(srcPC)
        DCP_ICP_PC.paint_uniform_color([0, 1, 0])
        DCP_ICP_TRANSFORM = ICPIter(DCP_ICP_PC, copy.deepcopy(targetPC), DCP_TRANSFORM)
        TRANSFORM_FINAL = DCP_ICP_TRANSFORM.transformation

        # DCP_PC = copy.deepcopy(srcPC)
        # DCP_PC.paint_uniform_color([1, 1, 0.4])
        # DCP_PC.transform(DCP_TRANSFORM)
        # o3d.visualization.draw_geometries([FGR_ICP_PC, targetPC, DCP_ICP_PC.transform(DCP_ICP_TRANS_FINAL), DCP_PC], 
        #                                   zoom=1,
        #                                   front=[1, 1, 1],
        #                                   lookat=[0, 0, 0],
        #                                   up=[0, 1, 0])
        
        
        #========Baseline ICP Method========
        # ICP_PC = copy.deepcopy(srcPC)
        # ICP_PC.paint_uniform_color([0, 1, 0])
        # viewer.add_geometry(ICP_PC)
        # iterSize = 50
        # for i in range(iterSize):
        #     ICP_TRANSFORM = ICPIter(ICP_PC, copy.deepcopy(targetPC), np.identity(4), iterSize = 1)
        #     ICP_PC.transform(ICP_TRANSFORM.transformation)
        #     viewer.update_geometry(ICP_PC)
        #     viewer.poll_events()
        #     viewer.update_renderer()
        
        # ICP_PC = copy.deepcopy(srcPC)
        # ICP_TRANSFORM = ICPIter(ICP_PC, copy.deepcopy(targetPC), np.identity(4))
        # TRANSFORM_FINAL = ICP_TRANSFORM.transformation
        
        
        
        ANGLE_LIST.append(Rotation.from_matrix(TRANSFORM_FINAL[:3, :3]).as_euler('zyx'))
        TRANS_LIST.append(TRANSFORM_FINAL[:3, 3])
        ANGLE_ROT_LIST.append(TRANSFORM_FINAL[:3, :3])

        print('\n')
        # print('ICPxyz: ', Rotation.from_matrix(DCP_ICP_TRANS_FINAL[:3, :3]).as_euler('xyz', True))
        print('ICPzyx: ', Rotation.from_matrix(TRANSFORM_FINAL[:3, :3]).as_euler('zyx', True))
        print('euler: ', np.degrees(euler_ab.numpy()))
        
        
        if (counter.Count() == True):
            break
    
    #========================Reformat the Losses========================
    rotations_ab = np.concatenate(rotations_ab, axis=0)
    translations_ab = np.concatenate(translations_ab, axis=0)
    rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
    translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)

    rotations_ba = np.concatenate(rotations_ba, axis=0)
    translations_ba = np.concatenate(translations_ba, axis=0)
    rotations_ba_pred = np.concatenate(rotations_ba_pred, axis=0)
    translations_ba_pred = np.concatenate(translations_ba_pred, axis=0)
    eulers_ab = np.concatenate(eulers_ab, axis=0)
    eulers_ba = np.concatenate(eulers_ba, axis=0)

    return total_loss * 1.0 / num_examples, \
           rotations_ab, translations_ab, rotations_ab_pred, translations_ab_pred, \
           rotations_ba, translations_ba, rotations_ba_pred, translations_ba_pred, \
           eulers_ab, eulers_ba, ANGLE_LIST, TRANS_LIST, ANGLE_ROT_LIST




def test(args, net, test_loader, textio):

    test_loss, \
    test_rotations_ab, test_translations_ab, test_rotations_ab_pred, test_translations_ab_pred, \
    test_rotations_ba, test_translations_ba, test_rotations_ba_pred, test_translations_ba_pred, \
    test_eulers_ab, test_eulers_ba, angleList, transList, ANGLE_ROT_LIST = test_one_epoch(args, net, test_loader)

    test_rotations_ab_pred_euler = npmat2euler(test_rotations_ab_pred)
    test_r_mse_ab = np.mean((test_rotations_ab_pred_euler - np.degrees(test_eulers_ab)) ** 2)
    test_r_rmse_ab = np.sqrt(test_r_mse_ab)
    test_r_mae_ab = np.mean(np.abs(test_rotations_ab_pred_euler - np.degrees(test_eulers_ab)))
    test_t_mse_ab = np.mean((test_translations_ab - test_translations_ab_pred) ** 2)
    test_t_rmse_ab = np.sqrt(test_t_mse_ab)
    test_t_mae_ab = np.mean(np.abs(test_translations_ab - test_translations_ab_pred))

    test_rotations_ba_pred_euler = npmat2euler(test_rotations_ba_pred, 'xyz')
    test_r_mse_ba = np.mean((test_rotations_ba_pred_euler - np.degrees(test_eulers_ba)) ** 2)
    test_r_rmse_ba = np.sqrt(test_r_mse_ba)
    test_r_mae_ba = np.mean(np.abs(test_rotations_ba_pred_euler - np.degrees(test_eulers_ba)))
    test_t_mse_ba = np.mean((test_translations_ba - test_translations_ba_pred) ** 2)
    test_t_rmse_ba = np.sqrt(test_t_mse_ba)
    test_t_mae_ba = np.mean(np.abs(test_translations_ba - test_translations_ba_pred))

    textio.cprint('==FINAL TEST==')
    textio.cprint('A--------->B')
    textio.cprint('EPOCH:: %d, Loss: %f, rot_MSE: %f, rot_RMSE: %f, '
                  'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                  % (-1, test_loss, \
                     test_r_mse_ab, test_r_rmse_ab, test_r_mae_ab, \
                     test_t_mse_ab, test_t_rmse_ab, test_t_mae_ab))
    textio.cprint('B--------->A')
    textio.cprint('EPOCH:: %d, Loss: %f, rot_MSE: %f, rot_RMSE: %f, '
                  'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                  % (-1, test_loss, \
                     test_r_mse_ba, test_r_rmse_ba, test_r_mae_ba, \
                     test_t_mse_ba, test_t_rmse_ba, test_t_mae_ba))
    
    avgAngRMSE = 0
    ranks = [0 for i in range(5)]# [0]: err<1, [1]: err<5, [2]: err<10, [3]: err<20, [4]: err<30
    print('Angle display in degrees: [Z, Y, X]')
    for i, ang in enumerate(angleList):
        # angDiff = np.abs(np.degrees(test_eulers_ab[i]) - np.degrees(ang))
        predAng = Rotation.from_matrix(ANGLE_ROT_LIST[i]).as_euler('zyx', True)
        GTAng = Rotation.from_matrix(test_rotations_ab[i]).as_euler('zyx', True)
        angDiff = np.abs(predAng - GTAng)
        AngRMSE = np.sqrt(np.mean(angDiff) ** 2)
        avgAngRMSE += AngRMSE
        print('Angle diff: ', angDiff, (' RMSE: %f' %AngRMSE))
        
        if (AngRMSE <= 1):
            ranks[0] += 1
        if (AngRMSE <= 5):
            ranks[1] += 1
        if (AngRMSE <= 10):
            ranks[2] += 1
        if (AngRMSE <= 20):
            ranks[3] += 1
        if (AngRMSE <= 30):
            ranks[4] += 1
    
    print('Average Angle Error: %f' %(avgAngRMSE / len(angleList)))
    for i, rk in enumerate(ranks):
        print('rank[%d]: %d' %(i, rk))
    
    
def main():
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dcp', metavar='N',
                        choices=['dcp'],
                        help='Model to use, [dcp]')
    parser.add_argument('--emb_nn', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Embedding nn to use, [pointnet, dgcnn]')
    parser.add_argument('--pointer', type=str, default='transformer', metavar='N',
                        choices=['identity', 'transformer'],
                        help='Attention-based pointer generator to use, [identity, transformer]')
    parser.add_argument('--head', type=str, default='svd', metavar='N',
                        choices=['mlp', 'svd', ],
                        help='Head to use, [mlp, svd]')
    parser.add_argument('--emb_dims', type=int, default=512, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--n_blocks', type=int, default=1, metavar='N',
                        help='Num of blocks of encoder&decoder')
    parser.add_argument('--n_heads', type=int, default=4, metavar='N',
                        help='Num of heads in multiheadedattention')
    parser.add_argument('--ff_dims', type=int, default=1024, metavar='N',
                        help='Num of dimensions of fc in transformer')
    parser.add_argument('--dropout', type=float, default=0.1, metavar='N',
                        help='Dropout ratio in transformer')
    parser.add_argument('--batch_size', type=int, default=1, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', action='store_true', default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--cudaF', action='store_true', default=True,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', action='store_true', default=True,
                        help='evaluate the model')
    parser.add_argument('--cycle', type=bool, default=False, metavar='N',
                        help='Whether to use cycle consistency')
    parser.add_argument('--gaussian_noise', type=bool, default=False, metavar='N',
                        help='Wheter to add gaussian noise')
    parser.add_argument('--unseen', type=bool, default=False, metavar='N',
                        help='Wheter to test on unseen category')
    parser.add_argument('--num_points', type=int, default=512, metavar='N',
                        help='Num of points to use')
    parser.add_argument('--dataset', type=str, default='modelnet40', choices=['modelnet40'], metavar='N',
                        help='dataset to use')
    parser.add_argument('--dataset_path', type=str, default='D:/Datasets/modelnet40_ply_hdf5_2048', choices=['D:/Datasets/modelnet40_ply_hdf5_2048', 'data/modelnet40_ply_hdf5_2048'], metavar='N',
                        help='dataset path')
    parser.add_argument('--viewF', type=bool, default=True, metavar='N',
                        help='Wheter to test on unseen category')
    parser.add_argument('--factor', type=float, default=4, metavar='N',
                        help='Divided factor for rotations')
    parser.add_argument('--model_path', type=str, default='pretrained/dcp_v1.t7', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N',
                        help='start position for model fine tune')

    args = parser.parse_args()
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    logPath = os.path.join('checkpoints', args.exp_name, 'run.log')
    if (not os.path.exists('checkpoints')):
        os.mkdir('checkpoints')
    if (not os.path.exists(os.path.join('checkpoints', args.exp_name))):
        os.mkdir(os.path.join('checkpoints', args.exp_name))
    textio = IOStream(logPath)
    textio.cprint(str(args))

    if args.dataset == 'modelnet40':
        # test_loader = DataLoader(ModelNet40H5(DIR_PATH = args.dataset_path, 
        #                                       templateNumber = args.num_points, 
        #                                       targetNumber = args.num_points, 
        #                                       dataPartition = 'test', targetGaussianNoise = args.gaussian_noise, 
        #                                       targetViewPC = args.viewF), 
        #                           batch_size = args.test_batch_size, 
        #                           shuffle=True, drop_last=False)
        test_loader = DataLoader(ModelNet40PCD(DIR_PATH = 'D:/Datasets/ModelNet40_VALID_512_2'))
    else:
        raise Exception("not implemented")
    
    if torch.cuda.is_available():
        net = DCP(args).cuda()
        net.load_state_dict(torch.load(args.model_path), strict=False)
    else:
        args.cudaF = False
        net = DCP(args)
        net.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')), strict=False)
        
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    test(args, net, test_loader, textio)


    print('FINISH')

if __name__ == '__main__':
    main()
