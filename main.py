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
from torchvision.models import mobilenet_v3_small
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

def dirichlet_split_noniid(train_labels, alpha, n_clients):
    '''
    按照参数为alpha的Dirichlet分布将样本索引集合划分为n_clients个子集
    '''
    n_classes = train_labels.max()+1
    # (K, N) 类别标签分布矩阵X，记录每个类别划分到每个client去的比例
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
    # (K, ...) 记录K个类别对应的样本索引集合
    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]

    # 记录N个client分别对应的样本索引集合
    client_idcs = [[] for _ in range(n_clients)]
    for k_idcs, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例fracs将类别为k的样本索引k_idcs划分为了N个子集
        # i表示第i个client，idcs表示其对应的样本索引集合idcs
        for i, idcs in enumerate(np.split(k_idcs,
                                          (np.cumsum(fracs)[:-1]*len(k_idcs)).
                                          astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs

#随机延迟
def generate_random(n):
    lower_bound = n * 0.8
    upper_bound = n * 1.2 
    return random.uniform(lower_bound, upper_bound)  

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

    avg_model = p_workers[0].model
    for param_name in avg_model:
        avg_model[param_name] = sum(worker.weight * worker.model[param_name] for worker in p_workers)
    for worker in p_workers:
        worker.model = avg_model

    #调整迭代次数为最新
    max_epoch = max(worker.iteration for worker in p_workers)
    for worker in p_workers:
        worker.iteration = max_epoch
    
def control( p_nums_set, update_times, queue, reduce_flag, decay_factor, delay):
    while True :
        p_workers = []
        while len(p_workers) < p_nums_set :
            model = queue.get()
            p_workers.append(model)
        DYN_model_agg(p_workers , decay_factor)
        t = generate_random(delay)
        time.sleep(t)
        for worker in p_workers :
            reduce_flag[worker.ID].put(worker)
        update_times[0] += 1

# 定义一个函数将单通道图像复制三次以创建三通道图像
def to_three_channels(x):
    return x.repeat(3, 1, 1)

def data_set(n):
    # 数据集预处理参数
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(to_three_channels)  # 将单通道图像复制三次以创建三通道图像
 
    ])
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)


    # transform_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])

    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])

    # train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    
    # test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)



    
    size = [1/n]*n

    trainsets = random_split(train_data, size)
    return trainsets, test_data

def data_set_asy(n):
    #数据集预处理参数
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(to_three_channels)  # 将单通道图像复制三次以创建三通道图像

    ])

    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # transform_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])

    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])

    # train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    
    # test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_labels = np.array(train_data.targets)

    client_idcs = dirichlet_split_noniid(train_labels, alpha=1, n_clients=n)

    trainsets= []
    for i in range(n):
        trainsets.append(Subset(train_data,client_idcs[i]))

    return trainsets, test_data



if __name__ == '__main__':
    #实验记录
    # run_times = 0
    worker_nums_set = int(sys.argv[1])
    p_nums_set = int(sys.argv[2])
    decay_factor = float(sys.argv[3])   
    delay = float(sys.argv[4])   
    print('The p_num is ', p_nums_set)
    print('The worker_num is ', worker_nums_set)
    print('The decay is ', decay_factor)
    print('The delay is ', delay)
    # for i in range (0, worker_nums_set):
    #     with open('record_acc%d.txt'%(i), 'a') as f:
    #         f.write('%d %d\n' %(worker_nums_set, p_nums_set))
    #     f.close()
        # with open('record_loss%d.txt'%(i), 'a') as f:
        #     f.write('%d %d\n' %(worker_nums_set, p_nums_set))
        # f.close()

    best_acc = 0  #所有节点的该轮最佳acc
    lowest_acc = 0 #所有节点的该轮最低acc
    flag = 0
    #分割数据集
    trainsets, testset = data_set(worker_nums_set)
    
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
        model = mobilenet_v3_small()
        #修改最后一层，以满足MNIST数据集的10个类别的要求
        model.classifier[-1] = nn.Linear(1024, 10)
        dataset = trainsets[worker_num]
        # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4) ##使用随机梯度下降法
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        # criterion = nn.CrossEntropyLoss()
        # scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        workers.append(Worker(worker_num, q, reduce_flag[worker_num], acc_list , model, dataset, testset, optimizer)) 
    Control = Process(target= control,args=( p_nums_set, update_times, q, reduce_flag, decay_factor, delay))

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
        if best_acc >= 95 and flag == 0:
            one_converge = update_times[0]
            flag = 1
        elif lowest_acc >= 95 and flag == 1:
            all_converge = update_times[0]
            break
        time.sleep(10)

    with open('record.txt', 'a') as f:
        f.write('\n%d %d %d %d %f' %(worker_nums_set, p_nums_set, one_converge, all_converge, best_acc))
    f.close()
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
from torchvision.models import mobilenet_v3_small
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

def dirichlet_split_noniid(train_labels, alpha, n_clients):
    '''
    按照参数为alpha的Dirichlet分布将样本索引集合划分为n_clients个子集
    '''
    n_classes = train_labels.max()+1
    # (K, N) 类别标签分布矩阵X，记录每个类别划分到每个client去的比例
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
    # (K, ...) 记录K个类别对应的样本索引集合
    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]

    # 记录N个client分别对应的样本索引集合
    client_idcs = [[] for _ in range(n_clients)]
    for k_idcs, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例fracs将类别为k的样本索引k_idcs划分为了N个子集
        # i表示第i个client，idcs表示其对应的样本索引集合idcs
        for i, idcs in enumerate(np.split(k_idcs,
                                          (np.cumsum(fracs)[:-1]*len(k_idcs)).
                                          astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs

#随机延迟
def generate_random(n):
    lower_bound = n * 0.8
    upper_bound = n * 1.2 
    return random.uniform(lower_bound, upper_bound)  

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

    avg_model = p_workers[0].model
    for param_name in avg_model:
        avg_model[param_name] = sum(worker.weight * worker.model[param_name] for worker in p_workers)
    for worker in p_workers:
        worker.model = avg_model

    #调整迭代次数为最新
    max_epoch = max(worker.iteration for worker in p_workers)
    for worker in p_workers:
        worker.iteration = max_epoch
    
def control( p_nums_set, update_times, queue, reduce_flag, decay_factor, delay):
    while True :
        p_workers = []
        while len(p_workers) < p_nums_set :
            model = queue.get()
            p_workers.append(model)
        DYN_model_agg(p_workers , decay_factor)
        t = generate_random(delay)
        time.sleep(t)
        for worker in p_workers :
            reduce_flag[worker.ID].put(worker)
        update_times[0] += 1

# 定义一个函数将单通道图像复制三次以创建三通道图像
def to_three_channels(x):
    return x.repeat(3, 1, 1)

def data_set(n):
    # 数据集预处理参数
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(to_three_channels)  # 将单通道图像复制三次以创建三通道图像
 
    ])
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)


    # transform_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])

    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])

    # train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    
    # test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)



    
    size = [1/n]*n

    trainsets = random_split(train_data, size)
    return trainsets, test_data

def data_set_asy(n):
    #数据集预处理参数
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(to_three_channels)  # 将单通道图像复制三次以创建三通道图像

    ])

    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # transform_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])

    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])

    # train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    
    # test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_labels = np.array(train_data.targets)

    client_idcs = dirichlet_split_noniid(train_labels, alpha=1, n_clients=n)

    trainsets= []
    for i in range(n):
        trainsets.append(Subset(train_data,client_idcs[i]))

    return trainsets, test_data



if __name__ == '__main__':
    #实验记录
    # run_times = 0
    worker_nums_set = int(sys.argv[1])
    p_nums_set = int(sys.argv[2])
    decay_factor = float(sys.argv[3])   
    delay = float(sys.argv[4])   
    print('The p_num is ', p_nums_set)
    print('The worker_num is ', worker_nums_set)
    print('The decay is ', decay_factor)
    print('The delay is ', delay)
    # for i in range (0, worker_nums_set):
    #     with open('record_acc%d.txt'%(i), 'a') as f:
    #         f.write('%d %d\n' %(worker_nums_set, p_nums_set))
    #     f.close()
        # with open('record_loss%d.txt'%(i), 'a') as f:
        #     f.write('%d %d\n' %(worker_nums_set, p_nums_set))
        # f.close()

    best_acc = 0  #所有节点的该轮最佳acc
    lowest_acc = 0 #所有节点的该轮最低acc
    flag = 0
    #分割数据集
    trainsets, testset = data_set(worker_nums_set)
    
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
        model = mobilenet_v3_small()
        #修改最后一层，以满足MNIST数据集的10个类别的要求
        model.classifier[-1] = nn.Linear(1024, 10)
        dataset = trainsets[worker_num]
        # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4) ##使用随机梯度下降法
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        # criterion = nn.CrossEntropyLoss()
        # scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        workers.append(Worker(worker_num, q, reduce_flag[worker_num], acc_list , model, dataset, testset, optimizer)) 
    Control = Process(target= control,args=( p_nums_set, update_times, q, reduce_flag, decay_factor, delay))

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
        if best_acc >= 95 and flag == 0:
            one_converge = update_times[0]
            flag = 1
        elif lowest_acc >= 95 and flag == 1:
            all_converge = update_times[0]
            break
        time.sleep(10)

    with open('record.txt', 'a') as f:
        f.write('\n%d %d %d %d %f' %(worker_nums_set, p_nums_set, one_converge, all_converge, best_acc))
    f.close()
