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
import csv
import os

DEG2RAD = 3.1415926 / 180.0

class Rigid():
    def __init__(self, rotation = 0, translation = 0, eulerAng = []):
        self.rotation = rotation
        self.translation = translation
        self.eulerAng = eulerAng
    
    def getRandomRigid(self, angleRange = 90, translationRange = 0.5):
        anglex = np.random.uniform(-angleRange, angleRange) * DEG2RAD
        angley = np.random.uniform(-angleRange, angleRange) * DEG2RAD
        anglez = np.random.uniform(-angleRange, angleRange) * DEG2RAD
        self.eulerAng = np.asarray([anglez, angley, anglex]).astype('float32')
        self.rotation = Rotation.from_euler('zyx', self.eulerAng).as_matrix().astype('float32')
        self.translation = np.array([np.random.uniform(-translationRange, translationRange), 
                                   np.random.uniform(-translationRange, translationRange), 
                                   np.random.uniform(-translationRange, translationRange)]).astype('float32')


    def getInvRigid(self):
        rotation_inv = self.rotation.T
        translation_inv = -rotation_inv.dot(self.translation)
        eulerAng_inv = -self.eulerAng[::-1]
        return Rigid(rotation_inv, translation_inv, eulerAng_inv)
    

class ModelNet40H5(Dataset):
    def __init__(self, DIR_PATH = 'D:/Datasets/modelnet40_ply_hdf5_2048', dataPartition = 'None', 
                 templateNumber = 1024, targetNumber = 1024, 
                 targetGaussianNoise = True, targetViewPC = False, 
                 angleRange = 90, translationRange = 0.5):
        
        self.data, self.label = self.load_data(DIR_PATH, dataPartition)
        self.number = templateNumber
        self.targetNumber = targetNumber
        self.factor = 4
        self.targetGaussianNoise = targetGaussianNoise
        self.viewF = targetViewPC
        self.angleRange = angleRange
        self.translationRange = translationRange
                
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

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, item):
        rigidAB = Rigid()
        rigidAB.getRandomRigid(self.angleRange, self.translationRange)
        rigidBA = rigidAB.getInvRigid()
        
        pc = self.data[item]
        if (self.viewF):
            pc_view, _ = GetRandomViewPointCloud(pc, self.targetNumber)
            pc1 = pc[:self.number].T
            pc2 = (rigidAB.rotation @ (pc_view.T)).T + rigidAB.translation
        else:
            pc1 = (pc[:self.number]).T
            pc2 = np.random.permutation(pc)
            pc2 = ((rigidAB.rotation @ pc2.T).T + rigidAB.translation)[:self.targetNumber]
        if self.targetGaussianNoise:
            pc2 = jitter_pointcloud(pc2)
        pc2 = np.random.permutation(pc2).T
        
        return pc1.astype('float32'), pc2.astype('float32'), \
            rigidAB.rotation, rigidAB.translation, rigidBA.rotation, rigidBA.translation, \
                rigidAB.eulerAng, rigidBA.eulerAng
        # return pc1.astype('float32'), pc2.astype('float32'), R_ab, translation_ab, R_ba, translation_ba, euler_ab, euler_ba
        # return pc2.astype('float32'), pc1.astype('float32'), R_ba, translation_ba, R_ab, translation_ab, euler_ba, euler_ab


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.05):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud

def GetRandomViewPointCloud(points, num = -1):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    cam_rho = np.linalg.norm(pcd.get_max_bound() - pcd.get_min_bound())
    cam_theta = np.random.uniform(-180, 180) * DEG2RAD
    cam_phi = np.random.uniform(0, 90) * DEG2RAD
    camPosition = [np.cos(cam_theta) * np.sin(cam_phi), np.sin(cam_theta) * np.sin(cam_phi), np.cos(cam_phi)]
    camPosition = [ i * cam_rho for i in camPosition]
    _, sub_points_map = pcd.hidden_point_removal(camPosition, cam_rho * 200)
    sub_pcd = pcd.select_by_index(sub_points_map)
    sub_points = np.asarray(sub_pcd.points)
    if (sub_points.shape[0] < 200):
        sub_points, camPosition = GetRandomViewPointCloud(points, num)
    if (num != -1):
        if (sub_points.shape[0] >= num):
            sub_points = sub_points[:num]
        else:
            while (sub_points.shape[0] < num):
                add_points = sub_points[:(num - sub_points.shape[0])]
                add_points = jitter_pointcloud(add_points)
                sub_points = np.concatenate((sub_points, add_points), axis = 0)
                sub_points = np.random.permutation(sub_points)
    return sub_points, camPosition


class ValidationModel():
    def __init__(self):
        self.number = 0
        self.templateModelList = []
        self.targetModelList = []
        self.template2TargetRigidList = []
        self.target2TemplateRigidList = []
    
    def addToModelQueue(self, templateModel, targetModel, rigidTemplate2Target, rigidTarget2Template):
        self.templateModelList.append(templateModel)
        self.targetModelList.append(targetModel)
        self.template2TargetRigidList.append(rigidTemplate2Target)
        self.target2TemplateRigidList.append(rigidTarget2Template)
        self.number += 1
    
    def writeModelToFile(self, DIR_PATH = 'ModelNet40_VALID', modelType = 'pcd'):
        csvFirstRow = ['TemplateModelName', 'TargetModelName', 
                       'Rotation_temp2tar', 'Translation_temp2tar', 
                       'Rotation_tar2temp', 'Translation_tar2temp', 
                       'EulerAngle_temp2tar', 'EulerAngle_tar2temp']
        
        # ========================Directory Initial Process========================
        storeTemplateDIR = os.path.join(DIR_PATH, 'template')
        storeTargetDIR = os.path.join(DIR_PATH, 'target')
        if not os.path.exists(DIR_PATH):
            os.makedirs(DIR_PATH)
        if not os.path.exists(storeTemplateDIR):
            os.makedirs(storeTemplateDIR)
        if not os.path.exists(storeTargetDIR):
            os.makedirs(storeTargetDIR)
        
        # ========================CSV File Process========================
        with open('%s/rigids.csv' %DIR_PATH, 'w', encoding = 'utf-8', newline = '') as fp:
            csvWriter = csv.writer(fp)
            csvWriter.writerow(csvFirstRow)
            for i, rigid in enumerate(zip(self.template2TargetRigidList, self.target2TemplateRigidList)):
                templateModelName = ('template_%d' %i)
                targetModelName = ('target_%d' %i)
                writeRow = [templateModelName, targetModelName, 
                            rigid[0].rotation, rigid[0].translation, 
                            rigid[1].rotation, rigid[1].translation, 
                            rigid[0].eulerAng, rigid[1].eulerAng]
                csvWriter.writerow(writeRow)
        
        # ========================Model File Process========================
        o3dTypeF = True if type(self.templateModelList[0]) == type(o3d.geometry.PointCloud()) else False
        for i, model in enumerate(zip(self.templateModelList, self.targetModelList)):
            templateModelName = ('template_%d.%s' %(i, modelType))
            targetModelName = ('target_%d.%s' %(i, modelType))
            if (o3dTypeF):
                o3d.io.write_point_cloud(os.path.join(storeTemplateDIR, templateModelName), model[0])
                o3d.io.write_point_cloud(os.path.join(storeTargetDIR, targetModelName), model[1])
            else:
                templatePC = o3d.geometry.PointCloud()
                templatePC.points = o3d.utility.Vector3dVector(model[0])
                targetPC = o3d.geometry.PointCloud()
                targetPC.points = o3d.utility.Vector3dVector(model[1])
                o3d.io.write_point_cloud(os.path.join(storeTemplateDIR, templateModelName), templatePC)
                o3d.io.write_point_cloud(os.path.join(storeTargetDIR, targetModelName), targetPC)
    
    def ReadModelFromFile(self, DIR_PATH = 'ModelNet40_VALID', modelType = 'pcd'):
        # ========================Read CSV File========================
        readTemplateDIR = os.path.join(DIR_PATH, 'template')
        readTargetDIR = os.path.join(DIR_PATH, 'target')
        with open('%s/rigids.csv' %DIR_PATH, 'r', encoding = 'utf-8', newline = '') as fp:
            csvReader = csv.reader(fp)
            for i, item in enumerate(csvReader):
                if (i == 0):# Ignore variable labels
                    continue
                templateModelPath = os.path.join(readTemplateDIR, ('%s.%s' %(item[0], modelType)))
                targetModelPath = os.path.join(readTargetDIR, ('%s.%s' %(item[1], modelType)))
                if ((not os.path.exists(templateModelPath)) or (not os.path.exists(targetModelPath))):
                    continue
                templateModel = o3d.io.read_point_cloud(templateModelPath)
                targetModel = o3d.io.read_point_cloud(targetModelPath)
                
                temp2TarRotStr = item[2].replace('[', '').replace(']', '').split()
                temp2TarRotList = [float(i) for i in temp2TarRotStr]
                temp2TarRot = np.array(temp2TarRotList).reshape((3, 3))
                temp2TarTransStr = item[3].strip('[]').split()
                temp2TarTransList = [float(i) for i in temp2TarTransStr]
                temp2TarTrans = np.array(temp2TarTransList)
                temp2TarEulerAngStr = item[6].strip('[]').split()
                temp2TarEulerAngList = [float(i) for i in temp2TarEulerAngStr]
                temp2TarEulerAng = np.array(temp2TarEulerAngList)
                
                tar2TempRotStr = item[4].replace('[', '').replace(']', '').split()
                tar2TempRotList = [float(i) for i in tar2TempRotStr]
                tar2TempRot = np.array(tar2TempRotList).reshape((3, 3))
                tar2TempTransStr = item[5].strip('[]').split()
                tar2TempTransList = [float(i) for i in tar2TempTransStr]
                tar2TempTrans = np.array(tar2TempTransList)
                tar2TempEulerAngStr = item[7].strip('[]').split()
                tar2TempEulerAngList = [float(i) for i in tar2TempEulerAngStr]
                tar2TempEulerAng = np.array(tar2TempEulerAngList)
                
                temp2TarRigid = Rigid(temp2TarRot, temp2TarTrans, temp2TarEulerAng)
                tar2TempRigid = Rigid(tar2TempRot, tar2TempTrans, tar2TempEulerAng)
                
                self.number += 1
                self.templateModelList.append(templateModel)
                self.targetModelList.append(targetModel)
                self.template2TargetRigidList.append(temp2TarRigid)
                self.target2TemplateRigidList.append(tar2TempRigid)

class ModelNet40PCD(Dataset):
    def __init__(self, DIR_PATH):
        self.models = ValidationModel()
        self.models.ReadModelFromFile(DIR_PATH = DIR_PATH)
        
    def __len__(self):
        return self.models.number
    
    def __getitem__(self, item):
        templateModel = self.models.templateModelList[item]
        targetModel = self.models.targetModelList[item]
        pc1 = np.asarray(templateModel.points).T
        pc2 = np.asarray(targetModel.points).T
        template2TargetRigid = self.models.template2TargetRigidList[item]
        target2TemplateRigid = self.models.target2TemplateRigidList[item]
        return pc1.astype('float32'), pc2.astype('float32'), \
            template2TargetRigid.rotation.astype('float32'), template2TargetRigid.translation.astype('float32'), \
                target2TemplateRigid.rotation.astype('float32'), target2TemplateRigid.translation.astype('float32'), \
                    template2TargetRigid.eulerAng.astype('float32'), target2TemplateRigid.eulerAng.astype('float32')


if __name__ == '__main__':
    # mod = ModelNet40H5(targetViewPC = True, dataPartition = 'test', 
    #                    templateNumber = 1024, targetNumber = 1024, 
    #                    angleRange = 90, translationRange = 0.5)
    # loader = DataLoader(mod)
    # storeModel = ValidationModel()
    
    # for cnt, i in enumerate(loader):
    #     # pts, pos = GetRandomViewPointCloud(i[0])
    #     pts0 = np.squeeze(i[0].numpy().T, axis = 2)
    #     pcd0 = o3d.geometry.PointCloud()
    #     pcd0.points = o3d.utility.Vector3dVector(pts0)
    #     pts1 = np.squeeze(i[1].numpy().T, axis = 2)
    #     pcd1 = o3d.geometry.PointCloud()
    #     pcd1.points = o3d.utility.Vector3dVector(pts1)
    #     # o3d.visualization.draw_geometries([pcd0, pcd1], 
    #     #                                   zoom=1,
    #     #                                   # front=pos,
    #     #                                   front=[1, 0, 0],
    #     #                                   lookat=[0, 0, 0],
    #     #                                   up=[0, 1, 0])
    #     rig0 = Rigid(np.squeeze(i[2].numpy()), np.squeeze(i[3].numpy()), np.squeeze(i[6].numpy()))
    #     rig1 = Rigid(np.squeeze(i[4].numpy()), np.squeeze(i[5].numpy()), np.squeeze(i[7].numpy()))
    #     storeModel.addToModelQueue(pcd0, pcd1, rig0, rig1)
    #     print(pcd0, pcd1)
    #     cnt += 1
    #     if (cnt >= 50):
    #         break
    # storeModel.writeModelToFile(DIR_PATH = 'D:/Datasets/ModelNet40_VALID_1024_2')
    readModel = ValidationModel()
    readModel.ReadModelFromFile(DIR_PATH = 'D:/Datasets/ModelNet40_VALID_1024_2')
    for pcd in zip(readModel.templateModelList, readModel.targetModelList):
        o3d.visualization.draw_geometries([pcd[0], pcd[1]], 
                                          zoom=1,
                                          front=[1, 0, 0],
                                          lookat=[0, 0, 0],
                                          up=[0, 1, 0])