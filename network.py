import torch
import torch.nn as nn

# 基础的残差模块
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, ch_in, ch_out, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch_out)
        # inplace为True，将计算得到的值直接覆盖之前的值,可以节省时间和内存
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.downsample = None
        if ch_out != ch_in:
            # 如果输入通道数和输出通道数不相同，使用1×1的卷积改变通道数
            self.downsample = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self,x):
        identity = x
        out = self.bn1(self.conv1(x))
        out = self.relu(out)
        out = self.bn2(self.conv2(out))
        if self.downsample != None:
            identity = self.downsample(x)
        out += identity
        relu = nn.ReLU()
        out = relu(out)
        return out

# 改进型的残差模块
class Bottleneck(nn.Module):
    expansion = 4  #扩展，即通道数为之前的4倍
    def __init__(self, ch_in, ch_out, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.conv3 = nn.Conv2d(ch_out, ch_out * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(ch_out * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if ch_in != ch_out * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(ch_in, ch_out * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(ch_out * self.expansion)
            )

    def forward(self, x):
        identity = x
        out = self.bn1(self.conv1(x))
        out = self.relu(out)
        out = self.bn2(self.conv2(out))
        out = self.relu(out)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        relu = nn.ReLU()
        out = relu(out)
        return out

# 实现ResNet网络
class ResNet(nn.Module):
    # 初始化；block：残差块结构；layers：残差块层数；num_classes：输出层神经元即分类数
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        # 改变后的通道数
        self.channel = 64
        # 第一个卷积层
        self.conv1 = nn.Conv2d(3, self.channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 残差网络的四个残差块堆
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接层，也是输出层
        self.fc = nn.Linear(512 * block.expansion, num_classes)



    # 用于堆叠残差块
    def _make_layer(self, block, ch_out, blocks, stride=1):
        layers = []
        layers.append(block(self.channel, ch_out, stride))
        self.channel = ch_out * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.channel, ch_out))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
 
 
class RNN(nn.Module):
 
    def __init__(self):
 
        super(RNN, self).__init__()
 
        self.rnn = nn.RNN(input_size=28, hidden_size=64, num_layers=1, batch_first=True)
 
        self.fc = nn.Linear(64, 10)
 
        
 
    def forward(self, x):
 
        out, h = self.rnn(x, None)
 
        out = self.fc(out[:, -1, :])
 
        return out


# ResNet18生成方法
def resnet18(num_classes=1000):
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes)
    return model

# # ResNet34生成方法
def ResNet34(num_classes=1000):
    model = ResNet(BasicBlock, [3,4,6,3], num_classes)
    return model

# ResNet50生成方法
def resnet50(num_classes=1000):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes)
    return model

