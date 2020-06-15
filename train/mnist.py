import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

# Customize module
import models


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, loader, optimizer, epoch):
    """Train model"""

    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        output = F.log_softmax(output, dim=1)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Epoch {}: [{:<5d}/{:<5d} ({:2.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(loader.dataset),
                100. * batch_idx / len(loader), loss.item()))
    print('\n')


def evaluate(model, device, loader, set_name):
    """Evaluate model"""

    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # Sum up batch loss
            # Get the index of the max log-probability
            total_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    total_loss /= len(loader.dataset)

    print('{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        set_name, total_loss, correct, len(loader.dataset),
        100. * correct / len(loader.dataset)))


def main(args):
    # Set GPUs
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.visible_gpus if use_cuda else ''

    # Set environment
    torch.manual_seed(args.seed)
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Get loader for training, validate, and testing set
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.data_dir, train=True, download=True,
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
        batch_size=args.test_batch_size, shuffle=True, **kwargs)



    # Get model then initial parameters
    # model = Net().to(device)

    NN_params = {'CNN_L2D1': [32, 64],
                 'CNN_L2D2': [64, 128],
                 'CNN_L4D1': [32, 64, 128, 256],
                 'CNN_L4D2': [64, 128, 256, 512]}
    model = models.ConvNet(conv_channels=NN_params[args.model_arch]).to(device)


    # Get optimizer and learning rate scheduler
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)



    # Train, validate, and testing
    for epoch in range(1, args.epochs+1):
        train(args, model, device, train_loader, optimizer, epoch)
        evaluate(model, device, test_loader, 'Test')
        scheduler.step()


    # Final evaluation on each set
    evaluate(model, device, train_loader, 'Train')
    evaluate(model, device, test_loader, 'Test')


    # Save trained model
    if args.save_model:
        torch.save(model.state_dict(),
                   os.path.join(args.exp_dir, 'mnist_cnn.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MNIST via PyTorch')

    # Environment settings
    parser.add_argument('--no-cuda', dest='no_cuda', default=False,
                        action='store_true', help='disables CUDA training')
    parser.add_argument('-vg', '--visible-gpus', dest='visible_gpus',
                        default='6', help='set cuda visible devices')
    parser.add_argument('--seed', dest='seed', default=286, type=int,
                        metavar='S', help='random seed (default: 286)')

    # Paths settings
    parser.add_argument('--data-dir', dest='data_dir', default='../../data',
                        metavar='PATH', help='data directory path')
    parser.add_argument('--exp-dir', dest='exp_dir', default='../../exp',
                        metavar='PATH', help='experiments home directory path')
    parser.add_argument('-n', '--exp-name', dest='exp_name',
                        help='experiment name')

    # Training settings
    parser.add_argument('--model-arch', dest='model_arch', default='CNN_L2D1',
                        help='choose model architecture')
    parser.add_argument('-bs', '--batch-size', dest='batch_size',
                        type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', dest='test_batch_size',
                        type=int, default=512, metavar='N',
                        help='input batch size for testing (default: 512)')
    parser.add_argument('-e', '--epochs', dest='epochs',
                        type=int, default=15, metavar='N',
                        help='number of epochs to train (default: 15)')
    parser.add_argument('-lr', '--learning_rate', dest='lr',
                        type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', dest='gamma', type=float, default=0.7,
                        metavar='M', help='lr step gamma (default: 0.7)')
    parser.add_argument('--log-interval', dest='log_interval',
                        type=int, default=10, metavar='N',
                        help='log training status after how many batches')
    parser.add_argument('--save-model', dest='save_model', default=False,
                        action='store_true', help='save current model or not')

    # Version
    parser.add_argument('--version', action='version', version='%(prog)s 0.1.0')
    args = parser.parse_args()

    # Set experiment directory
    args.exp_dir = os.path.join(args.exp_dir, args.exp_name)
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)

    print('Start experiment under %s with model %s' %
          (args.exp_dir, args.model_arch))
    main(args)

