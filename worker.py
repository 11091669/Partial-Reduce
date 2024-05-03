import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torchvision import datasets, transforms,models
from torch.utils.data import DataLoader
from multiprocessing import Process, Queue
import time
import random
from copy import deepcopy

#传递规约模型以及迭代信息
class reduce_Model():
    def __init__(self, model, ID):
        self._ID = ID                   #节点ID
        self._model = model.state_dict() #节点模型
        self._iteration = 0             #工作节点迭代数
        self._r_iteration = 0           #工作节点相对迭代数
        self._weight = 0                #聚合权重 

    @property
    def ID(self):
        return self._ID
    @ID.setter
    def ID(self, ID):
        self._ID = ID

    @property
    def model(self):
        return self._model
    @model.setter
    def model(self, model):
        self._model = model

    @property
    def iteration(self):
        return self._iteration
    @iteration.setter
    def iteration(self, iteration):
        self._iteration = iteration
    
    @property
    def r_iteration(self):
        return self._r_iteration
    @r_iteration.setter
    def r_iteration(self, r_iteration):
        self._r_iteration = r_iteration
    
    @property
    def weight(self):
        return self._weight
    @weight.setter
    def weight(self, weight):
        self._weight = weight


#工作线程
class Worker(Process):
    def __init__(self, ID 
                 , queue
                 , reduce
                 , acc_list
                 , model
                 , dataset
                 , testset
                 , optimizer 
                 , scheduler = None
                 , criterion = nn.CrossEntropyLoss()):
        torch.cuda.set_device(ID % torch.cuda.device_count())
        super(Process,self).__init__()
        self._ID = ID
        self._model = model.cuda()
        self._reduce_model = reduce_Model(model, ID)
        self._queue = queue             #传递计算好的模型
        self._reduce = reduce           #接收规约后的模型
        self._dataset = dataset         #数据集
        self._testset = testset         #测试集
        self._dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
        self._testloader = DataLoader(testset, batch_size=256)
        self._optimizer = optimizer     #模型优化器
        self._scheduler = scheduler     #模型步长优化
        self._criterion = criterion     #损失函数
        self._acc_list = acc_list       #节点测试准确率


    def run(self):
        self.train(self._queue,self._reduce)

    def rand_error(self):  ##随机产生训练延迟，构造异构环境
        
        # 生成一个随机数，范围在0到1之间
        random_value = random.random()

        # 根据概率分配返回结果
        if random_value < 0.6:
            return 
        elif random_value < 0.8:
            time.sleep(1)
            return
        elif random_value < 0.95:
            time.sleep(2)
            return
        else:
            time.sleep(3)
            return 

    # Training
    def train(self, queue, reduce):
        torch.cuda.set_device(self._ID % torch.cuda.device_count())
        
        for epoch in range(3000):
            print('worker %d epoch %d' %(self._reduce_model.ID, epoch))
            self.test()
            #切换为训练模式
            self.model.cuda()
            self.model.train()
            for batch_idx, (inputs, targets) in enumerate(self._dataloader):
                #训练流程
                inputs = inputs.cuda()
                targets = targets.cuda()
                self._optimizer.zero_grad()
                # inputs = inputs.view(-1, 28, 28)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self._optimizer.step()
                #迭代次数加一
                self._reduce_model.iteration += 1

                #构造异构
                self.rand_error()

                #加载模型到cpu进行聚合
                device = torch.device('cpu')
                corrupt_model = deepcopy(self.model).to(device)
                self._reduce_model.model = corrupt_model.state_dict()
                queue.put(self._reduce_model)

                #阻塞等待信号
                reduce_model = reduce.get()
                self._reduce_model.model = reduce_model.model
                self._reduce_model.iteration = reduce_model.iteration
            
                #将接收的模型加载到训练模型
                self.model.load_state_dict(reduce_model.model)
    
    def test(self):
        torch.cuda.set_device(self._ID % torch.cuda.device_count())
        self.model.cuda()
        self.model.eval()
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self._testloader):
            inputs = inputs.cuda()
            targets = targets.cuda()
            # inputs = inputs.view(-1, 28, 28)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        acc = 100. * correct / total
        self._acc_list[self._reduce_model.ID] = acc
        with open('record_acc%d.txt'%(self._ID), 'a') as f:
            f.write('%f\n'%(acc))
        f.close()
        # self.scheduler.step()

    @property
    def model(self):
        return self._model
    @model.setter
    def model(self, model):
        self._model = model

    @property
    def optimizer(self):
        return self._optimizer
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @property
    def testloader(self):
        return self._testloader
    @testloader.setter
    def testloader(self, testloader):
        self._testloader = testloader


    @property
    def reduce_flag(self):
        return self._reduce_flag
    @reduce_flag.setter
    def reduce_flag(self, reduce_flag):
        self._reduce_flag = reduce_flag

    @property
    def scheduler(self):
        return self._scheduler
    @scheduler.setter
    def scheduler(self, scheduler):
        self._scheduler = scheduler

    @property
    def criterion(self):
        return self._criterion
    @criterion.setter
    def criterion(self, criterion):
        self._criterion = criterion
    
    @property
    def acc(self):
        return self._acc
    @acc.setter
    def acc(self, acc):
        self._acc = acc

    @property
    def dataloader(self):
        return self._dataloader
    @dataloader.setter
    def dataloader(self, dataloader):
        self._dataloader = dataloader

