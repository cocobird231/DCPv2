# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 20:53:45 2021

@author: User
"""

import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation
import open3d as o3d


DEG2RAD = 3.1415926 / 180.0

class Rigid():
    def __init__(self, rotation = 0, translation = 0, eulerAng = []):
        self.rotation = rotation
        self.translation = translation
        self.eulerAng = eulerAng
    
    def getRandomRigid(self):
        anglex = np.random.uniform(-90, 90) * DEG2RAD
        angley = np.random.uniform(-90, 90) * DEG2RAD
        anglez = np.random.uniform(-90, 90) * DEG2RAD
        self.eulerAng = np.asarray([anglez, angley, anglex]).astype('float32')
        self.rotation = Rotation.from_euler('zyx', self.eulerAng).as_matrix().astype('float32')
        self.translation = np.array([np.random.uniform(-0.5, 0.5), 
                                   np.random.uniform(-0.5, 0.5), 
                                   np.random.uniform(-0.5, 0.5)]).astype('float32')


    def getInvRigid(self):
        rotation_inv = self.rotation.T
        translation_inv = -rotation_inv.dot(self.translation)
        eulerAng_inv = -self.eulerAng[::-1]
        return Rigid(rotation_inv, translation_inv, eulerAng_inv)
    

class ModelNet40H5(Dataset):
    def __init__(self, DIR_PATH = 'D:/Datasets/modelnet40_ply_hdf5_2048', dataPartition = 'None', 
                 templateNumber = 1024, targetNumber = 1024, 
                 targetGaussianNoise = True, targetViewPC = False):
        
        self.data, self.label = self.load_data(DIR_PATH, dataPartition)
        self.number = templateNumber
        self.targetNumber = targetNumber
        self.factor = 4
        self.targetGaussianNoise = targetGaussianNoise
        self.viewF = targetViewPC
                
    def load_data(self, DIR_PATH, dataPartition):
        all_data = []
        all_label = []
        dataNamePattern = '/ply_data*.h5'
        if (dataPartition != 'None'):
            dataNamePattern = ('/ply_data_%s*.h5' %dataPartition)
        print(dataNamePattern)
        for h5_name in glob.glob(DIR_PATH + dataNamePattern):
            f = h5py.File(h5_name)
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            f.close()
            all_data.append(data)
            all_label.append(label)
        all_data = np.concatenate(all_data, axis=0)
        all_label = np.concatenate(all_label, axis=0)
        return all_data, all_label

    def jitter_pointcloud(self, pointcloud, sigma=0.01, clip=0.05):
        N, C = pointcloud.shape
        pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
        return pointcloud
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, item):
        
        # anglex = np.random.uniform(-90, 90) * DEG2RAD
        # angley = np.random.uniform(-90, 90) * DEG2RAD
        # anglez = np.random.uniform(-90, 90) * DEG2RAD
        # euler_ab = np.asarray([anglez, angley, anglex]).astype('float32')
        # euler_ba = -euler_ab[::-1]
        
        # rotation_ab = Rotation.from_euler('zyx', euler_ab)
        # R_ab = rotation_ab.as_matrix().astype('float32')
        # R_ba = R_ab.T
        
        # translation_ab = np.array([np.random.uniform(-0.5, 0.5), 
        #                            np.random.uniform(-0.5, 0.5), 
        #                            np.random.uniform(-0.5, 0.5)]).astype('float32')
        # translation_ba = -R_ba.dot(translation_ab)
        rigidAB = Rigid()
        rigidAB.getRandomRigid()
        rigidBA = rigidAB.getInvRigid()
        
        pc = self.data[item]
        if (self.viewF):
            pc_view, _ = GetRandomViewPointCloud(pc)
            pc1 = pc[:pc_view.shape[0]].T
            pc2 = (rigidAB.rotation @ (pc_view.T)).T + rigidAB.translation
        else:
            pc1 = (pc[:self.number]).T
            pc2 = np.random.permutation(pc)
            pc2 = ((rigidAB.rotation @ pc2.T).T + rigidAB.translation)[:self.targetNumber]
        if self.targetGaussianNoise:
            pc2 = self.jitter_pointcloud(pc2)
        pc2 = np.random.permutation(pc2).T
        
        # pc = self.data[item]
        # pc1 = 0
        # pc2 = 0
        # if (self.viewF):
        #     pc_view, _ = GetRandomViewPointCloud(pc)
        #     pc1 = pc[:pc_view.shape[0]].T
        #     pc2 = (R_ab @ (pc_view.T)).T + translation_ab
        # else:
        #     pc1 = (pc[:self.number]).T
        #     pc2 = np.random.permutation(pc)
        #     pc2 = ((R_ab @ pc2.T).T + translation_ab)[:self.targetNumber]
        # if self.targetGaussianNoise:
        #     pc2 = self.jitter_pointcloud(pc2)
        # pc2 = np.random.permutation(pc2).T
        
        return pc1.astype('float32'), pc2.astype('float32'), \
            rigidAB.rotation, rigidAB.translation, rigidBA.rotation, rigidBA.translation, \
                rigidAB.eulerAng, rigidBA.eulerAng
        # return pc1.astype('float32'), pc2.astype('float32'), R_ab, translation_ab, R_ba, translation_ba, euler_ab, euler_ba
        # return pc2.astype('float32'), pc1.astype('float32'), R_ba, translation_ba, R_ab, translation_ab, euler_ba, euler_ab


def GetRandomViewPointCloud(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    cam_rho = np.linalg.norm(pcd.get_max_bound() - pcd.get_min_bound())
    cam_theta = np.random.uniform(-180, 180) * DEG2RAD
    cam_phi = np.random.uniform(0, 179) * DEG2RAD
    camPosition = [np.cos(cam_theta) * np.sin(cam_phi), np.sin(cam_theta) * np.sin(cam_phi), np.cos(cam_phi)]
    camPosition = [ i * cam_rho for i in camPosition]
    _, sub_points_map = pcd.hidden_point_removal(camPosition, cam_rho * 200)
    sub_pcd = pcd.select_by_index(sub_points_map)
    sub_points = np.asarray(sub_pcd.points)
    if (sub_points.shape[0] < 64):
        sub_points, camPosition = GetRandomViewPointCloud(points)
    return sub_points, camPosition
        
if __name__ == '__main__':
    mod = ModelNet40H5(targetViewPC = True)
    loader = DataLoader(mod)
    cnt = 0
    for i in loader:
        # pts, pos = GetRandomViewPointCloud(i[0])
        pts = np.squeeze(i[1].numpy().T, axis = 2)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        o3d.visualization.draw_geometries([pcd], 
                                          zoom=1,
                                          # front=pos,
                                          front=[1, 0, 0],
                                          lookat=[0, 0, 0],
                                          up=[0, 1, 0])

        cnt += 1
        if (cnt > 10):
            break
    