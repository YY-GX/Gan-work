# # ---------------------------------------------------------------------------- #
# # An implementation of https://arxiv.org/pdf/1512.03385.pdf                    #
# # See section 4.2 for the model architecture on CIFAR-10                       #
# # Some part of the code was referenced from below                              #
# # https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py   #
# # ---------------------------------------------------------------------------- #

# import torch
# import torch.nn as nn
# import torchvision
# import torchvision.transforms as transforms


# # Device configuration

# device = torch.device('cuda:{}'.format(torch.cuda.device_count() - 1) if torch.cuda.is_available() else 'cpu')
# # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # device = torch.device('cpu')

# print(device)

# # Hyper-parameters
# num_epochs = 888888888
# batch_size = 1
# learning_rate = 0.001

# # Image preprocessing modules
# transform = transforms.Compose([
#     transforms.Pad(4),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomCrop(32),
#     transforms.ToTensor()])

# # CIFAR-10 dataset
# train_dataset = torchvision.datasets.CIFAR10(root='../../data/',
#                                              train=True, 
#                                              transform=transform,
#                                              download=True)

# test_dataset = torchvision.datasets.CIFAR10(root='../../data/',
#                                             train=False, 
#                                             transform=transforms.ToTensor())

# # Data loader
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                            batch_size=batch_size,
#                                            shuffle=True)

# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                           batch_size=batch_size,
#                                           shuffle=False)

# # 3x3 convolution
# def conv3x3(in_channels, out_channels, stride=1):
#     return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
#                      stride=stride, padding=1, bias=False)

# # Residual block
# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1, downsample=None):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = conv3x3(in_channels, out_channels, stride)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(out_channels, out_channels)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.downsample = downsample
        
#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         if self.downsample:
#             residual = self.downsample(x)
#         out += residual
#         out = self.relu(out)
#         return out

# # ResNet
# class ResNet(nn.Module):
#     def __init__(self, block, layers, num_classes=10):
#         super(ResNet, self).__init__()
#         self.in_channels = 16
#         self.conv = conv3x3(3, 16)
#         self.bn = nn.BatchNorm2d(16)
#         self.relu = nn.ReLU(inplace=True)
#         self.layer1 = self.make_layer(block, 16, layers[0])
#         self.layer2 = self.make_layer(block, 32, layers[1], 2)
#         self.layer3 = self.make_layer(block, 64, layers[2], 2)
#         self.avg_pool = nn.AvgPool2d(8)
#         self.fc = nn.Linear(64, num_classes)
        
#     def make_layer(self, block, out_channels, blocks, stride=1):
#         downsample = None
#         if (stride != 1) or (self.in_channels != out_channels):
#             downsample = nn.Sequential(
#                 conv3x3(self.in_channels, out_channels, stride=stride),
#                 nn.BatchNorm2d(out_channels))
#         layers = []
#         layers.append(block(self.in_channels, out_channels, stride, downsample))
#         self.in_channels = out_channels
#         for i in range(1, blocks):
#             layers.append(block(out_channels, out_channels))
#         return nn.Sequential(*layers)
    
#     def forward(self, x):
#         out = self.conv(x)
#         out = self.bn(out)
#         out = self.relu(out)
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.avg_pool(out)
#         out = out.view(out.size(0), -1)
#         out = self.fc(out)
#         return out
    
# model = ResNet(ResidualBlock, [2, 2, 2]).to(device)


# # Loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# # For updating learning rate
# def update_lr(optimizer, lr):    
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

# # Train the model
# total_step = len(train_loader)
# curr_lr = learning_rate
# for epoch in range(num_epochs):
#     for i, (images, labels) in enumerate(train_loader):
#         images = images.to(device)
#         labels = labels.to(device)
        
#         # Forward pass
#         outputs = model(images)
#         loss = criterion(outputs, labels)
        
#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         if (i+1) % 100 == 0:
#             print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
#                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

#     # Decay learning rate
#     if (epoch+1) % 20 == 0:
#         curr_lr /= 3
#         update_lr(optimizer, curr_lr)

# # Test the model
# model.eval()
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in test_loader:
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

#     print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

# # Save the model checkpoint
# torch.save(model.state_dict(), 'resnet.ckpt')


# from __future__ import print_function
import argparse
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.optim.lr_scheduler import StepLR


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
        
#         self.conv1 = nn.Conv2d(1, 32, 3, 1)
#         self.conv2 = nn.Conv2d(32, 64, 3, 1)
#         self.dropout1 = nn.Dropout2d(0.25)
#         self.dropout2 = nn.Dropout2d(0.5)
#         self.fc1 = nn.Linear(9216, 128)
#         self.fc2 = nn.Linear(128, 10)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, 2)
#         x = self.dropout1(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.dropout2(x)
#         x = self.fc2(x)
#         output = F.log_softmax(x, dim=1)
#         return output


# def train(args, model, device, train_loader, optimizer, epoch):
#     model.train()
    
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = F.nll_loss(output, target)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % args.log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))


# def test(args, model, device, test_loader):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
#             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#             correct += pred.eq(target.view_as(pred)).sum().item()

#     test_loss /= len(test_loader.dataset)

#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))


# def main():
#     # Training settings
#     parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
#     parser.add_argument('--batch-size', type=int, default=1, metavar='N',
#                         help='input batch size for training (default: 64)')
#     parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',
#                         help='input batch size for testing (default: 1000)')
#     parser.add_argument('--epochs', type=int, default=888888888, metavar='N',
#                         help='number of epochs to train (default: 14)')
#     parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
#                         help='learning rate (default: 1.0)')
#     parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
#                         help='Learning rate step gamma (default: 0.7)')
#     parser.add_argument('--no-cuda', action='store_true', default=False,
#                         help='disables CUDA training')
#     parser.add_argument('--seed', type=int, default=1, metavar='S',
#                         help='random seed (default: 1)')
#     parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                         help='how many batches to wait before logging training status')

#     parser.add_argument('--save-model', action='store_true', default=False,
#                         help='For Saving the current Model')
#     args = parser.parse_args()
#     use_cuda = not args.no_cuda and torch.cuda.is_available()

#     torch.manual_seed(args.seed)

# #     device = torch.device("cuda" if use_cuda else "cpu")
#     device = torch.device('cuda:{}'.format(torch.cuda.device_count() - 1) if torch.cuda.is_available() else 'cpu')

#     kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
#     train_loader = torch.utils.data.DataLoader(
#         datasets.MNIST('../data', train=True, download=True,
#                        transform=transforms.Compose([
#                            transforms.ToTensor(),
#                            transforms.Normalize((0.1307,), (0.3081,))
#                        ])),
#         batch_size=args.batch_size, shuffle=True, **kwargs)
#     test_loader = torch.utils.data.DataLoader(
#         datasets.MNIST('../data', train=False, transform=transforms.Compose([
#                            transforms.ToTensor(),
#                            transforms.Normalize((0.1307,), (0.3081,))
#                        ])),
#         batch_size=args.test_batch_size, shuffle=True, **kwargs)

#     model = Net().to(device)
#     optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

#     scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
#     for epoch in range(1, args.epochs + 1):
#         train(args, model, device, train_loader, optimizer, epoch)
#         test(args, model, device, test_loader)
#         scheduler.step()

#     if args.save_model:
#         torch.save(model.state_dict(), "mnist_cnn.pt")


# if __name__ == '__main__':
#     main()









# Code in file autograd/two_layer_net_autograd.py
import torch

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--gpuid', type=int, default=torch.cuda.device_count() - 1, metavar='N',
                        help='input batch size for training (default: 64)')
args = parser.parse_args()
gpuid = args.gpuid


# device = torch.device('cpu')
device = torch.device('cuda:{}'.format(gpuid) if torch.cuda.is_available() else 'cpu')
# device = torch.device('cuda') # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 2, 10, 1, 1

# Create random Tensors to hold input and outputs
x = torch.randn(N, D_in, device=device)
y = torch.randn(N, D_out, device=device)

# Create random Tensors for weights; setting requires_grad=True means that we
# want to compute gradients for these Tensors during the backward pass.
w1 = torch.randn(D_in, H, device=device, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, requires_grad=True)

learning_rate = 1e-6
for t in range(888888888888):
  # Forward pass: compute predicted y using operations on Tensors. Since w1 and
  # w2 have requires_grad=True, operations involving these Tensors will cause
  # PyTorch to build a computational graph, allowing automatic computation of
  # gradients. Since we are no longer implementing the backward pass by hand we
  # don't need to keep references to intermediate values.
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
  
  # Compute and print loss. Loss is a Tensor of shape (), and loss.item()
  # is a Python number giving its value.
    loss = (y_pred - y).pow(2).sum()
#     print(t, loss.item())

  # Use autograd to compute the backward pass. This call will compute the
  # gradient of loss with respect to all Tensors with requires_grad=True.
  # After this call w1.grad and w2.grad will be Tensors holding the gradient
  # of the loss with respect to w1 and w2 respectively.
    loss.backward()

  # Update weights using gradient descent. For this step we just want to mutate
  # the values of w1 and w2 in-place; we don't want to build up a computational
  # graph for the update steps, so we use the torch.no_grad() context manager
  # to prevent PyTorch from building a computational graph for the updates
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

    # Manually zero the gradients after running the backward pass
        w1.grad.zero_()
        w2.grad.zero_()
