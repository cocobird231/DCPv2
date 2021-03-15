# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 16:25:09 2021

@author: User
"""
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
from My_utilV2 import transform_point_cloud, npmat2euler
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from My_ReadCADV2 import ModelNet40H5

class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def _init_(args):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
    os.system('cp main.py checkpoints' + '/' + args.exp_name + '/' + 'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')


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

    for src, target, rotation_ab, translation_ab, rotation_ba, translation_ba, euler_ab, euler_ba in tqdm(test_loader):
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

        #transformed_src = transform_point_cloud(src, rotation_ab_pred, translation_ab_pred)

        #transformed_target = transform_point_cloud(target, rotation_ba_pred, translation_ba_pred)

        ###########################
        identity = torch.eye(3).cuda().unsqueeze(0).repeat(batch_size, 1, 1)
        loss = F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) \
               + F.mse_loss(translation_ab_pred, translation_ab)

        total_loss += loss.item() * batch_size


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
            rotations_ba, translations_ba, rotations_ba_pred, translations_ba_pred, eulers_ab, eulers_ba


def train_one_epoch(args, net, train_loader, opt):
    net.train()

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

    for src, target, rotation_ab, translation_ab, rotation_ba, translation_ba, euler_ab, euler_ba in tqdm(train_loader):
        src = src.cuda()
        target = target.cuda()
        rotation_ab = rotation_ab.cuda()
        translation_ab = translation_ab.cuda()
        rotation_ba = rotation_ba.cuda()
        translation_ba = translation_ba.cuda()

        batch_size = src.size(0)
        opt.zero_grad()
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

        #transformed_src = transform_point_cloud(src, rotation_ab_pred, translation_ab_pred)

        #transformed_target = transform_point_cloud(target, rotation_ba_pred, translation_ba_pred)
        ###########################
        identity = torch.eye(3).cuda().unsqueeze(0).repeat(batch_size, 1, 1)
        loss = F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) \
               + F.mse_loss(translation_ab_pred, translation_ab)

        loss.backward()
        opt.step()
        total_loss += loss.item() * batch_size

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
            rotations_ba, translations_ba, rotations_ba_pred, translations_ba_pred, eulers_ab, eulers_ba


def test(args, net, test_loader, boardio, textio):

    test_loss, \
    test_rotations_ab, test_translations_ab, \
    test_rotations_ab_pred, \
    test_translations_ab_pred, test_rotations_ba, test_translations_ba, test_rotations_ba_pred, \
    test_translations_ba_pred, test_eulers_ab, test_eulers_ba = test_one_epoch(args, net, test_loader)

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


def train(args, net, train_loader, test_loader, boardio, textio):
    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(net.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = MultiStepLR(opt, milestones=[75, 150, 200], gamma=0.1)


    best_test_loss = np.inf

    best_test_r_mse_ab = np.inf
    best_test_r_rmse_ab = np.inf
    best_test_r_mae_ab = np.inf
    best_test_t_mse_ab = np.inf
    best_test_t_rmse_ab = np.inf
    best_test_t_mae_ab = np.inf


    best_test_r_mse_ba = np.inf
    best_test_r_rmse_ba = np.inf
    best_test_r_mae_ba = np.inf
    best_test_t_mse_ba = np.inf
    best_test_t_rmse_ba = np.inf
    best_test_t_mae_ba = np.inf

    for epoch in range(args.start_epoch, args.epochs):
        train_loss, \
        train_rotations_ab, train_translations_ab, train_rotations_ab_pred, train_translations_ab_pred, \
        train_rotations_ba, train_translations_ba, train_rotations_ba_pred, train_translations_ba_pred, \
        train_eulers_ab, train_eulers_ba = train_one_epoch(args, net, train_loader, opt)
        
        test_loss, \
        test_rotations_ab, test_translations_ab, test_rotations_ab_pred, test_translations_ab_pred, \
        test_rotations_ba, test_translations_ba, test_rotations_ba_pred, test_translations_ba_pred, \
        test_eulers_ab, test_eulers_ba = test_one_epoch(args, net, test_loader)
        
        scheduler.step()

        train_rotations_ab_pred_euler = npmat2euler(train_rotations_ab_pred)
        train_r_mse_ab = np.mean((train_rotations_ab_pred_euler - np.degrees(train_eulers_ab)) ** 2)
        train_r_rmse_ab = np.sqrt(train_r_mse_ab)
        train_r_mae_ab = np.mean(np.abs(train_rotations_ab_pred_euler - np.degrees(train_eulers_ab)))
        train_t_mse_ab = np.mean((train_translations_ab - train_translations_ab_pred) ** 2)
        train_t_rmse_ab = np.sqrt(train_t_mse_ab)
        train_t_mae_ab = np.mean(np.abs(train_translations_ab - train_translations_ab_pred))

        train_rotations_ba_pred_euler = npmat2euler(train_rotations_ba_pred, 'xyz')
        train_r_mse_ba = np.mean((train_rotations_ba_pred_euler - np.degrees(train_eulers_ba)) ** 2)
        train_r_rmse_ba = np.sqrt(train_r_mse_ba)
        train_r_mae_ba = np.mean(np.abs(train_rotations_ba_pred_euler - np.degrees(train_eulers_ba)))
        train_t_mse_ba = np.mean((train_translations_ba - train_translations_ba_pred) ** 2)
        train_t_rmse_ba = np.sqrt(train_t_mse_ba)
        train_t_mae_ba = np.mean(np.abs(train_translations_ba - train_translations_ba_pred))

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

        if best_test_loss >= test_loss:
            best_test_loss = test_loss

            best_test_r_mse_ab = test_r_mse_ab
            best_test_r_rmse_ab = test_r_rmse_ab
            best_test_r_mae_ab = test_r_mae_ab

            best_test_t_mse_ab = test_t_mse_ab
            best_test_t_rmse_ab = test_t_rmse_ab
            best_test_t_mae_ab = test_t_mae_ab

            best_test_r_mse_ba = test_r_mse_ba
            best_test_r_rmse_ba = test_r_rmse_ba
            best_test_r_mae_ba = test_r_mae_ba

            best_test_t_mse_ba = test_t_mse_ba
            best_test_t_rmse_ba = test_t_rmse_ba
            best_test_t_mae_ba = test_t_mae_ba

            if torch.cuda.device_count() > 1:
                torch.save(net.module.state_dict(), 'checkpoints/%s/models/model.best.t7' % args.exp_name)
            else:
                torch.save(net.state_dict(), 'checkpoints/%s/models/model.best.t7' % args.exp_name)
        
        ############TEXT_IO
        textio.cprint('==TRAIN==')
        textio.cprint('A--------->B')
        textio.cprint('EPOCH:: %d, Loss: %f, rot_MSE: %f, rot_RMSE: %f, '
                      'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                      % (epoch, train_loss, \
                         train_r_mse_ab, train_r_rmse_ab, train_r_mae_ab, \
                         train_t_mse_ab, train_t_rmse_ab, train_t_mae_ab))
        
        textio.cprint('B--------->A')
        textio.cprint('EPOCH:: %d, Loss: %f, rot_MSE: %f, rot_RMSE: %f, '
                      'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                      % (epoch, train_loss, \
                         train_r_mse_ba, train_r_rmse_ba, train_r_mae_ba, \
                         train_t_mse_ba, train_t_rmse_ba, train_t_mae_ba))

        textio.cprint('==TEST==')
        textio.cprint('A--------->B')
        textio.cprint('EPOCH:: %d, Loss: %f, rot_MSE: %f, rot_RMSE: %f, '
                      'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                      % (epoch, test_loss, \
                         test_r_mse_ab, test_r_rmse_ab, test_r_mae_ab, \
                         test_t_mse_ab, test_t_rmse_ab, test_t_mae_ab))
        
        textio.cprint('B--------->A')
        textio.cprint('EPOCH:: %d, Loss: %f, rot_MSE: %f, rot_RMSE: %f, '
                      'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                      % (epoch, test_loss, \
                         test_r_mse_ba, test_r_rmse_ba, test_r_mae_ba, \
                         test_t_mse_ba, test_t_rmse_ba, test_t_mae_ba))

        textio.cprint('==BEST TEST==')
        textio.cprint('A--------->B')
        textio.cprint('EPOCH:: %d, Loss: %f, rot_MSE: %f, rot_RMSE: %f, '
                      'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                      % (epoch, best_test_loss, \
                         best_test_r_mse_ab, best_test_r_rmse_ab, best_test_r_mae_ab, \
                         best_test_t_mse_ab, best_test_t_rmse_ab, best_test_t_mae_ab))
        textio.cprint('B--------->A')
        textio.cprint('EPOCH:: %d, Loss: %f, rot_MSE: %f, rot_RMSE: %f, '
                      'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                      % (epoch, best_test_loss, \
                         best_test_r_mse_ba, best_test_r_rmse_ba, best_test_r_mae_ba, \
                         best_test_t_mse_ba, best_test_t_rmse_ba, best_test_t_mae_ba))
        
        ############TRAIN
        boardio.add_scalar('A-B/train/loss', train_loss, epoch)
        boardio.add_scalar('A-B/train/rotation/MSE', train_r_mse_ab, epoch)
        boardio.add_scalar('A-B/train/rotation/RMSE', train_r_rmse_ab, epoch)
        boardio.add_scalar('A-B/train/rotation/MAE', train_r_mae_ab, epoch)
        boardio.add_scalar('A-B/train/translation/MSE', train_t_mse_ab, epoch)
        boardio.add_scalar('A-B/train/translation/RMSE', train_t_rmse_ab, epoch)
        boardio.add_scalar('A-B/train/translation/MAE', train_t_mae_ab, epoch)

        boardio.add_scalar('B-A/train/loss', train_loss, epoch)
        boardio.add_scalar('B-A/train/rotation/MSE', train_r_mse_ba, epoch)
        boardio.add_scalar('B-A/train/rotation/RMSE', train_r_rmse_ba, epoch)
        boardio.add_scalar('B-A/train/rotation/MAE', train_r_mae_ba, epoch)
        boardio.add_scalar('B-A/train/translation/MSE', train_t_mse_ba, epoch)
        boardio.add_scalar('B-A/train/translation/RMSE', train_t_rmse_ba, epoch)
        boardio.add_scalar('B-A/train/translation/MAE', train_t_mae_ba, epoch)

        ############TEST
        boardio.add_scalar('A-B/test/loss', test_loss, epoch)
        boardio.add_scalar('A-B/test/rotation/MSE', test_r_mse_ab, epoch)
        boardio.add_scalar('A-B/test/rotation/RMSE', test_r_rmse_ab, epoch)
        boardio.add_scalar('A-B/test/rotation/MAE', test_r_mae_ab, epoch)
        boardio.add_scalar('A-B/test/translation/MSE', test_t_mse_ab, epoch)
        boardio.add_scalar('A-B/test/translation/RMSE', test_t_rmse_ab, epoch)
        boardio.add_scalar('A-B/test/translation/MAE', test_t_mae_ab, epoch)

        boardio.add_scalar('B-A/test/loss', test_loss, epoch)
        boardio.add_scalar('B-A/test/rotation/MSE', test_r_mse_ba, epoch)
        boardio.add_scalar('B-A/test/rotation/RMSE', test_r_rmse_ba, epoch)
        boardio.add_scalar('B-A/test/rotation/MAE', test_r_mae_ba, epoch)
        boardio.add_scalar('B-A/test/translation/MSE', test_t_mse_ba, epoch)
        boardio.add_scalar('B-A/test/translation/RMSE', test_t_rmse_ba, epoch)
        boardio.add_scalar('B-A/test/translation/MAE', test_t_mae_ba, epoch)

        ############BEST TEST
        boardio.add_scalar('A-B/best_test/loss', best_test_loss, epoch)
        boardio.add_scalar('A-B/best_test/rotation/MSE', best_test_r_mse_ab, epoch)
        boardio.add_scalar('A-B/best_test/rotation/RMSE', best_test_r_rmse_ab, epoch)
        boardio.add_scalar('A-B/best_test/rotation/MAE', best_test_r_mae_ab, epoch)
        boardio.add_scalar('A-B/best_test/translation/MSE', best_test_t_mse_ab, epoch)
        boardio.add_scalar('A-B/best_test/translation/RMSE', best_test_t_rmse_ab, epoch)
        boardio.add_scalar('A-B/best_test/translation/MAE', best_test_t_mae_ab, epoch)

        boardio.add_scalar('B-A/best_test/loss', best_test_loss, epoch)
        boardio.add_scalar('B-A/best_test/rotation/MSE', best_test_r_mse_ba, epoch)
        boardio.add_scalar('B-A/best_test/rotation/RMSE', best_test_r_rmse_ba, epoch)
        boardio.add_scalar('B-A/best_test/rotation/MAE', best_test_r_mae_ba, epoch)
        boardio.add_scalar('B-A/best_test/translation/MSE', best_test_t_mse_ba, epoch)
        boardio.add_scalar('B-A/best_test/translation/RMSE', best_test_t_rmse_ba, epoch)
        boardio.add_scalar('B-A/best_test/translation/MAE', best_test_t_mae_ba, epoch)

        if torch.cuda.device_count() > 1:
            torch.save(net.module.state_dict(), 'checkpoints/%s/models/model.%d.t7' % (args.exp_name, epoch))
        else:
            torch.save(net.state_dict(), 'checkpoints/%s/models/model.%d.t7' % (args.exp_name, epoch))
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dcp', metavar='N',
                        choices=['dcp'],
                        help='Model to use, [dcp]')
    parser.add_argument('--emb_nn', type=str, default='pointnet', metavar='N',
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
    parser.add_argument('--batch_size', type=int, default=20, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=10, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', action='store_true', default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--cudaF', type=bool, default=True,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluate the model')
    parser.add_argument('--cycle', type=bool, default=False, metavar='N',
                        help='Whether to use cycle consistency')
    parser.add_argument('--gaussian_noise', type=bool, default=True, metavar='N',
                        help='Wheter to add gaussian noise')
    parser.add_argument('--unseen', type=bool, default=False, metavar='N',
                        help='Wheter to test on unseen category')
    parser.add_argument('--num_points', type=int, default=1024, metavar='N',
                        help='Num of points to use')
    parser.add_argument('--dataset', type=str, default='modelnet40', choices=['modelnet40'], metavar='N',
                        help='dataset to use')
    parser.add_argument('--dataset_path', type=str, default='D:/Datasets/modelnet40_ply_hdf5_2048', choices=['D:/Datasets/modelnet40_ply_hdf5_2048', 'data/modelnet40_ply_hdf5_2048'], metavar='N',
                        help='dataset path')
    parser.add_argument('--factor', type=float, default=4, metavar='N',
                        help='Divided factor for rotations')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N',
                        help='Pretrained model path')

    args = parser.parse_args()
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    boardio = SummaryWriter(log_dir='checkpoints/' + args.exp_name)
    _init_(args)

    textio = IOStream('checkpoints/' + args.exp_name + '/run.log')
    textio.cprint(str(args))

    if args.dataset == 'modelnet40':
        train_loader = DataLoader(ModelNet40H5(DIR_PATH = args.dataset_path, 
                                               templateNumber = args.num_points, 
                                               targetNumber = args.num_points, 
                                               dataPartition = 'train', targetGaussianNoise = args.gaussian_noise), 
                                  batch_size = args.batch_size, 
                                  shuffle=True, drop_last=True)
        test_loader = DataLoader(ModelNet40H5(DIR_PATH = args.dataset_path, 
                                              templateNumber = args.num_points, 
                                              targetNumber = args.num_points, 
                                              dataPartition = 'test', targetGaussianNoise = args.gaussian_noise), 
                                 batch_size = args.test_batch_size, 
                                 shuffle=True, drop_last=False)
    else:
        raise Exception("not implemented")

    if args.model == 'dcp':
        net = DCP(args).cuda()
        if args.eval:
            if args.model_path == '':
                model_path = 'checkpoints' + '/' + args.exp_name + '/models/model.best.t7'
            else:
                model_path = args.model_path
                print(model_path)
            if not os.path.exists(model_path):
                print("can't find pretrained model")
                return
            net.load_state_dict(torch.load(model_path), strict=False)
        if args.model_path != '':
            model_path = args.model_path
            print(model_path)
            net.load_state_dict(torch.load(model_path), strict=False)
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
            print("Let's use", torch.cuda.device_count(), "GPUs!")
    else:
        raise Exception('Not implemented')
    if args.eval:
        test(args, net, test_loader, boardio, textio)
    else:
        train(args, net, train_loader, test_loader, boardio, textio)


    print('FINISH')
    boardio.close()


if __name__ == '__main__':
    main()
