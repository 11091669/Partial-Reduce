import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torchvision import datasets, transforms,models
from torch.utils.data import DataLoader
import os
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from network import RNN,resnet50
import numpy as np
import random
import sys
import time
from worker import Worker
from torch.utils.data import random_split
from multiprocessing import Process,Queue,set_start_method,Manager
import torch.multiprocessing as mp
from copy import deepcopy
from torch.utils.data import Subset

# 模型聚合
def calculate_weights(k, decay_factor=0.9):
    # 初始化权重
    weights = np.zeros(k + 1)
    # 计算EMA权重
    for i in range(1, k + 1):
        # 使用EMA公式计算权重
        weights[i] = ((1 - decay_factor)*decay_factor**(k-i))/(1 - decay_factor**k)

    # 归一化权重
    weights = weights / np.sum(weights)
    return weights
 
#动态p_reduce
def DYN_model_agg(p_workers, decay_factor):
    device = torch.device('cpu')
    # for worker in p_workers:
    #     worker.model.to('cuda:0')
    #计算模型权重
    min_iteration = min(worker.iteration for worker in p_workers)
    k = 0
    for worker in p_workers:
        worker.r_iteration = worker.iteration - min_iteration + 1
        if worker.r_iteration > k : k = worker.r_iteration
    weights = calculate_weights(k, decay_factor)

    p_workers = sorted(p_workers, key=lambda worker: worker.r_iteration)
    
    # 对模型的参数根据权重进行计算
    for worker in p_workers:
        worker.weight = 0
    for i in range(1, k + 1):
        count = 0
        for worker in p_workers:
            if worker.r_iteration == i:
                count += 1
        #相同相对迭代次数的模型平分权重
        if count > 0:
            for worker in p_workers:
                if worker.r_iteration == i: 
                    worker.weight += weights[i]/count
        #不存在该相对迭代次数的模型用第一个模型替代
        else:
            p_workers[0].weight +=  weights[i] 

    # # 将参数放回每个模型 
    # worker_weights = [] #记录模型权重
    # for worker in p_workers:
    #     worker_weights.append(worker.weight)

    avg_model = p_workers[0].model
    # for params in zip(*[worker.model.parameters() for worker in p_workers]):
    #     #每个模型的参数和对应权重相乘再加和
    #     param_avg = sum(weight * param.data for param, weight in zip(params, worker_weights))
    #     for param in params:
    #         param.data.copy_(param_avg)
    for param_name in avg_model:
        avg_model[param_name] = sum(worker.weight * worker.model[param_name] for worker in p_workers)
    for worker in p_workers:
        worker.model = avg_model

    #调整迭代次数为最新
    max_epoch = max(worker.iteration for worker in p_workers)
    for worker in p_workers:
        worker.iteration = max_epoch
    
def control( p_nums_set, update_times, queue, reduce_flag, decay_factor):
    while True :
        p_workers = []
        while len(p_workers) < p_nums_set :
            model = queue.get()
            p_workers.append(model)
        DYN_model_agg(p_workers , decay_factor)
        for worker in p_workers :
            reduce_flag[worker.ID].put(worker)
        update_times[0] += 1

def data_set(n):
    #数据集预处理参数
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,))
 
    # ])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    
    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)


    # train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    size = [1/n]*n

    trainsets = random_split(train_data, size)
    return trainsets, test_data

def data_set_asy(n):
    #数据集预处理参数
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,))
    # ])

    # train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    
    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)


    target_labels = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [2, 3, 4, 5, 6], [7, 8, 9, 0 ,1], [ 4, 5, 6, 7, 8], [ 0, 1, 2, 3, 9], [0, 6, 7, 8, 9], [5, 1, 2, 3, 4]]
    
    
    trainsets = []
    for x in range(0, n):
        selected_train_indices = []
        for i in range(len(train_data)):
            img, label = train_data[i]
            if label in target_labels[x]:
                selected_train_indices.append(i)
        trainsets.append(Subset(train_data, selected_train_indices))

    return trainsets, test_data



if __name__ == '__main__':
    #实验记录
    # run_times = 0
    worker_nums_set = int(sys.argv[1])
    p_nums_set = int(sys.argv[2])
    decay_factor = float(sys.argv[3])   
    print('The p_num is ', p_nums_set)
    print('The worker_num is ', worker_nums_set)
    print('The decay is ', decay_factor)
    for i in range (0, worker_nums_set):
        with open('record_acc%d.txt'%(i), 'a') as f:
            f.write('%d %d\n' %(worker_nums_set, p_nums_set))
        f.close()
        # with open('record_loss%d.txt'%(i), 'a') as f:
        #     f.write('%d %d\n' %(worker_nums_set, p_nums_set))
        # f.close()

    best_acc = 0  #所有节点的该轮最佳acc
    lowest_acc = 0 #所有节点的该轮最低acc
    flag = 0
    #分割数据集
    trainsets, testset = data_set_asy(worker_nums_set)
 
    set_start_method('spawn')
    
    #创建进程
    man = Manager()
    reduce_flag = man.list()    #放入规约好的模型
    acc_list = man.list()       #记录每个模型的ACC
    for i in range(worker_nums_set):
        reduce_flag.append(man.Queue()) #每个模型对应一个队列用来传输规约后模型
        acc_list.append(0)
    update_times = man.list() #更新次数
    update_times.append(0)
    one_converge = 0 #最好worker收敛的更新次数
    all_converge = 0 #所有worker收敛的更新次数
    
    workers = []
    # 初始化节点
    q = Queue()
    for worker_num in range(worker_nums_set):
        model = resnet50(10)
        dataset = trainsets[worker_num]
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4) ##使用随机梯度下降法
        criterion = nn.CrossEntropyLoss()
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        workers.append(Worker(worker_num, q, reduce_flag[worker_num], acc_list , model, dataset, testset, optimizer, scheduler, criterion)) 
    Control = Process(target= control,args=( p_nums_set, update_times, q, reduce_flag, decay_factor))

    #开始进程
    for i in range(worker_nums_set):
        workers[i].daemon = True
        workers[i].start()
    Control.daemon = True
    Control.start()
    #检查acc
    last_acc = 0
    while True :
        best_acc = max(acc_list)
        lowest_acc = min(acc_list)    
        if last_acc != best_acc:
            print(best_acc)
            last_acc = best_acc
        #设置收敛阈值
        if best_acc >= 70 and flag == 0:
            one_converge = update_times[0]
            flag = 1
        elif lowest_acc >= 70 and flag == 1:
            all_converge = update_times[0]
            break
        time.sleep(10)

    with open('record.txt', 'a') as f:
        f.write('\n%d %d %d %d %f' %(worker_nums_set, p_nums_set, one_converge, all_converge, best_acc))
    f.close()
