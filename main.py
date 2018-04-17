import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchnet as tnt

# Settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--dropout', type=float, default=0.25, metavar='P',
                    help='dropout probability (default: 0.25)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='heavy ball momentum in gradient descent (default: 0.9)')
parser.add_argument('--data-dir', type=str, default='./data',metavar='DIR')
args = parser.parse_args()
args.cuda =  torch.cuda.is_available()

# Print out arguments to the log
print('Training LeNet on MNIST')
for p in vars(args).items():
    print('  ',p[0]+': ',p[1])
print('\n')


# Data loaders
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(args.data_dir, train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(args.data_dir, train=False,download=True, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=1000, shuffle=True, **kwargs)


# The LeNet architecture, with dropout and batch normalization
class View(nn.Module):
    def __init__(self,o):
        super(View, self).__init__()
        self.o = o
    def forward(self,x):
        return x.view(-1, self.o)
    
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        def convbn(ci,co,ksz,psz,p):
            return nn.Sequential(
                nn.Conv2d(ci,co,ksz),
                nn.BatchNorm2d(co),
                nn.ReLU(True),
                nn.MaxPool2d(psz,stride=psz),
                nn.Dropout(p))

        self.m = nn.Sequential(
            convbn(1,20,5,3,args.dropout),
            convbn(20,50,5,2,args.dropout),
            View(50*2*2),
            nn.Linear(50*2*2, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(True),
            nn.Dropout(args.dropout),
            nn.Linear(500,10))

    def forward(self, x):
        return self.m(x)


# Initialize the model, the loss function and the optimizer
model = LeNet()
loss_function = nn.CrossEntropyLoss()
if args.cuda:
    model.cuda()
    loss_function.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum = args.momentum)


# Function to train the model on one epoch of data
def train(epoch):
    model.train()
    for batch_ix, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        if batch_ix % 100 == 0 and batch_ix>0:
            print('[Epoch %2d, batch %3d] training loss: %.4f' %
                (epoch, batch_ix, loss.data[0]))

            
# Test the model on one epoch of validation data 
def test():
    model.eval()
    test_loss = tnt.meter.AverageValueMeter()
    top1 = tnt.meter.ClassErrorMeter()
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        loss = loss_function(output, target)

        top1.add(output.data, target.data)
        test_loss.add(loss.data[0])

    print('[Epoch %2d] Average test loss: %.3f, accuracy: %.2f%%\n'
        %(epoch, test_loss.value()[0], top1.value()[0]))


if __name__=="__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test()
