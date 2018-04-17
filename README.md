In this guide we’ll learn how to build and train a deep neural network,
using Python with PyTorch. In the process we’ll also touch on Git, the
ubiquitous version control system for code development, and some other
basic command line utilities.

This guide is intended for people with no experience developing in a
command line environment.

# Installation and set up

## SSH

Typically, we’ll be running our code remotely on powerful machines with
many GPUs. To log in, we’ll be using the command `ssh`. Mac OSX and
Linux both come with SSH already installed. However if you are using
Windows, you’ll need to install something like
[PuTTY](https://www.putty.org/).

After a while, you’ll grow tired of entering your username and password.
Consider generating a `ssh` keypair, for example by following [this
tutorial](https://www.digitalocean.com/community/tutorials/how-to-set-up-ssh-keys--2).

### A basic example

Open up a terminal. SSH’ing in is easy:

``` bash
$ ssh -l your_username hostname.LotsOfGpews.ca
```

Now you’re sitting in the home directory of your remote GPU box.

## Python

We’ll need an installation of Python. Many Linux distros already have a
version of Python installed, but we will be using Anaconda. Anaconda’s
main appeal is it comes with a complete set of commonly used scientific
computing libraries.

Installation instructions for your particular operating system are found
[here](https://conda.io/docs/user-guide/install/index.html).

### Installation on a server

Once you’ve SSH’d into your GPU box, you may install Anaconda in your
home directory as follows.

Use the command `wget` to get the installation file into your home
folder.

``` bash
$ wget https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh
```

Then, run the installer.

``` bash
$ bash Anaconda3-5.1.0-Linux-x86_64.sh
```

Of course, the version numbers of the installer will change. Copy and
paste the link to the latest version from [the Anaconda installation
site](https://www.anaconda.com/download/#linux).

## PyTorch

The three biggest DNN libraries in Python are PyTorch, TensorFlow, and
Keras. We’ll be using [PyTorch](http://pytorch.org/). Head over to their
website and follow their installation instructions.

The installation depends on which version of CUDA (the proprietary
library for Nvidia GPUs) is installed on your system, if at all. In
Linux, one possible way to check the CUDA version is to issue the
command

``` bash
$ cat /usr/local/cuda/version.txt
```

## Other Python libraries

Some other useful Python libraries you may want to install are OpenCV
and TorchNet. To install [OpenCV](https://opencv.org/), issue the
command

``` bash
$ conda install -c menpo opencv
```

To install [TorchNet for PyTorch](https://github.com/pytorch/tnt), issue
the command

``` bash
$ pip install git+https://github.com/pytorch/tnt.git@master
```

## Git

Almost all modern software development uses some sort of version control
system, a means of managing edits to source code. We’ll be using Git,
the de facto standard in the open source world and the DNN community.

Most Linux distributions already have Git installed. If you are using
Mac OSX or Windows, go to the [Git website](https://git-scm.com/) and
follow the installation instructions.

We will be hosting the code we write on a central Git server (think
Dropbox for code), called a *repository*. The two biggest providers are
BitBucket and GitHub. We will use [GitHub](https://github.com/) – head
over and create an account.

Keeping your code on a central Git server will ease the synchonization
of code between your personal computer and your GPU box. A typical
workflow is to write and debug code on your personal computer (which may
not have any GPUs), and then run code on your GPU box. You will be
editing code on both machines, and may be sharing code with others as
well. Without a central Git repository, the code base will soon become a
mess, with different versions strewn about haphazardly. A central Git
repository alleviates this nightmare.

There are many other virtues of using Git, which will become more
apparent once you gain some experience. Once you’ve started using Git,
it’s hard to imagine writing any code without it.

# A working example

In this section we’ll write code to train a neural net for classifying
handwritten digits in the MNIST dataset \[1\], using the LeNet
architecture \[2\].

Using your favourite text editor *on your personal computer* write this
code to a file called `main.py`

We’ll first need to import all the required Python modules (analagous to
a library in C), which we’ll be using later.

``` python numberLines
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchnet as tnt
```

Let’s write our script so it can take in command line arguments, using
the module `argparse`. We’ll print out the arguments to the log, in case
we ever need to know how the script was called. Users can query these
command line arguments by calling `python ./main.py --help`.

``` python numberLines
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
```

Next, we write code for retrieving the training images and the test
images. The objects `train_loader` and `test_loader` are Python
generators: they may be iterated over in a `for` loop. At each
iteration, they return a *batch* of images.

``` python numberLines
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(args.data_dir, train=True, 
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(args.data_dir, train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=1000, shuffle=True, **kwargs)
```

Now define the DNN architecture, or model. For the MNIST dataset, we’ll
use LeNet with dropout \[3\] and batch normalization \[4\]. In PyTorch,
models are implemented using the class `torch.nn.Module`. A `Module` can
be made of nested `Module`s, which is convenient because most of the
standed NN layers have already been implemented. It’s up to you to put
them together.

Every `Module` must have a `forward` function, which is what is called
when the model is evaluated.

``` python numberLines
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
```

Next we’ll initialize an instance of the model, create a loss function
(objective function), and initialize an optimizer. PyTorch already has
some basic optimizers, in the `torch.optim` module. We’ll use the old
chesnut SGD, with a learning rate (step size) of 0.1 and with heavy ball
momentum.

``` python numberLines
model = LeNet()
loss_function = nn.CrossEntropyLoss()
if args.cuda:
    model.cuda()
    loss_function.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum = args.momentum)
```

Now we define a function to train one *epoch* of training data. An epoch
is simply one run through your training set. So if we were using full
gradients, an epoch would be one gradient step.

The function first puts the model in training mode (turns off dropout
and fixes the batch normalization constants). It then copies the data to
the GPU if CUDA is available. Next it wraps the data in a `Variable`
class, which is necessary for running PyTorch’s automatic
differentiation machine (to compute the gradient). It then zeros the
gradient buffer in the optimizer, calculates the loss value, and then
computes the gradient by calling `backward()` on the loss. It then takes
one gradient descent step. Finally, it reports the training loss every
100 steps.

``` python numberLines
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
```

Next, we define a function for evaluating the model on the test data,
which we’ll run after every training epoch.

``` python numberLines
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
```

Finally, we actually execute the code.

``` python numberLines
if __name__=="__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test()
```

## Running the code

If you have installed Anaconda and PyTorch on your own personal
computer, you can run (and debug) the code via

``` bash
$ python ./main.py
```

## Getting the code onto your GPU box

We’ll use Git to transfer this code onto your GPU box. First we’ll
create a Git repository on your local machine.

``` bash
$ cd /path/to/your/script
$ git init
```

Add the file and make your first commit.

    $ git add main.py
    $ git commit -m 'first commit'

Now go to your GitHub page and create a new repository. (Note that by
default new GitHub repositories are publicly available\!) Copy the URL
to the newly created remote repository. Next, on your local machine, add
the remote repository and push the changes from your machine to the
GitHub repository.

``` bash
$ git remote add origin git@github.com:your_username/your_new_repo.git
$ git push origin
```

You can now browse your code on your GitHub page.

Now, `ssh` into your GPU machine, and pull the code from the remote
repository. Then run it\!

``` bash
(GPU-box) $ git clone git@github.com:your_username/your_new_repo.git
(GPU-box) $ cd your_new_repo/
(GPU-box) $ python ./main.py
```

It’s sometimes convenient to write `bash` scripts to run your code. For
example if you don’t want to hog all the GPUs on your GPU box (PyTorch’s
default), or if you want to write the output to a log file. With this in
mind, let’s write the following `bash` script to the file `run.sh`,
using a text editor *on your GPU box*. The easiest command line editor
is Nano, but more seasoned veterans use either Vim or Emacs.

``` sh
#!/bin/bash

# An script to train LeNet on MNIST with SGD

LOGDIR='./logs/'
DATADIR='./data/'
mkdir -p $LOGDIR
mkdir -p $DATADIR
GPUID=0 # Select a GPU. If you want say two GPUs, set GPUID="0,1"

CUDA_VISIBLE_DEVICES=$GPUID \
python -u ../main.py\
  --data $DATADIR\
  --batch-size 128\
  --dropout 0.3\
  --momentum 0.9\
  --epochs 10 > $LOGDIR/log.out
```

Now make the script executable and run it in the background.

``` bash
(GPU-box) $ chmod +x run.sh
(GPU-box) $ ./run.sh &
```

That’s it. For a long run you can log off your GPU box and go for lunch.
If you want to watch the log file during the run, you can use `tail`:

``` bash
(GPU-box) $ tail -f logs/log.out
```

At this point, our script is not being tracked by Git. Suppose we want
to share our code with a collaborator, and have them run the same
script. Add the script and push it back to the remote GitHub repository.

``` bash
(GPU-box) $ git add run.sh
(GPU-box) $ git commit -m 'bash script to run main.py'
(GPU-box) $ git push origin
```

Now on your local computer, you can pull the changes you’ve made:

``` bash
$ git pull origin
```

And the file will magically appear.

# Tips and Tricks

There are a couple command line utilities you may use to make developing
in a command line environment less painful.

## Tmux

SSH by itself is fairly rudimentary. You only have access to one shell –
what if you want to open multiple shells on your remote machine? For
example, say you want to edit code in one shell, run code in another,
and monitor an active log file in a third shell. Do you SSH in to your
remote machine three times?

Thankfully no. Use `tmux`, a terminal multiplier. It allows you to open
as many shells as you want, all from one SSH log in.

But this isn’t even the main appeal of `tmux`. It’s main selling point
is that it allows for a persistent working state on remote machines. Say
your SSH connection is dropped (spotty WiFi perhaps). If you are using
only SSH, all your work will be lost – when you SSH back in, you’ll be
looking at a blank prompt. If instead, you were working in a `tmux`
session, all you have to do upon logging back in is issue the command

``` bash
$ tmux attach
```

and your session reappears.

There are about a million `tmux` tutorials out there, but
[here’s](https://danielmiessler.com/study/tmux/) one I like.

## `rsync`, `scp`

Git is extremely useful for syncronizing code, but you should *never*
use it for transferring data. Instead, use the commands `scp` and
`rsync`. Use `scp` for transferring individual files, and `rsync` for
syncronizing entire folders.

# Exersises

1.  Modify `main.py` to log the value of the loss functon in a ‘csv’
    file after each step of the optimizer. Use one ‘csv’ for training
    values, and another for test values.
2.  Write a Python script to import the csv file and plot it. For
    import, use the Python library `pandas`; for plotting use
    `matplotlib`.
3.  Create a class `schedule` in `main.py` implementing a learning rate
    scheduler. The class should have a function `step()`, which
    decreases the learning rate according to the schedule. Call
    `schedule.step()` after each epoch of test data.
4.  Using the Python library `pickle`, store the parameter values of the
    LeNet model after each epoch in a file. Compute the true gradient of
    the model at each parameter value, and compare the true gradient
    against the mini batch gradients.

# Citations

<div id="refs" class="references">

<div id="ref-mnist">

\[1\] Y. LeCun, “The MNIST database of handwritten digits,”
*http://yann.lecun.com/exdb/mnist/*.

</div>

<div id="ref-lenet">

\[2\] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, “Gradient-based
learning applied to document recognition,” *Proceedings of the IEEE*,
vol. 86, no. 11, pp. 2278–2324, 1998.

</div>

<div id="ref-dropout">

\[3\] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R.
Salakhutdinov, “Dropout: A simple way to prevent neural networks from
overfitting,” *The Journal of Machine Learning Research*, vol. 15, no.
1, pp. 1929–1958, 2014.

</div>

<div id="ref-batchnorm">

\[4\] S. Ioffe and C. Szegedy, “Batch normalization: Accelerating deep
network training by reducing internal covariate shift,” *arXiv preprint
arXiv:1502.03167*, 2015.

</div>

</div>
