import time
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from dataset import CheXpertDataset
# import pickle
import PIL.Image as Image


class LeNet(nn.Module):
    def __init__(self, channel=3, hideen=768, num_classes=10):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(hideen, num_classes)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def weights_init(m):
    try:
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.weight' % m._get_name())
    try:
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.bias' % m._get_name())


class ImageDataset(Dataset):
    def __init__(self, imgs, labs, transform=None):
        self.imgs = imgs  # img paths
        self.labs = labs  # labs is ndarray
        self.transform = transform
        del imgs, labs

    def __len__(self):
        return self.labs.shape[0]

    def __getitem__(self, idx):
        lab = self.labs[idx]
        img = Image.open(self.imgs[idx])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = self.transform(img)
        return img, lab


def lfw_dataset(lfw_path, shape_img):
    images_all = []
    labels_all = []
    folders = os.listdir(lfw_path)
    for foldidx, fold in enumerate(folders):
        files = os.listdir(os.path.join(lfw_path, fold))
        for f in files:
            if len(f) > 4 and f[-4:] == '.jpg':
                images_all.append(os.path.join(lfw_path, fold, f))
                labels_all.append(foldidx)

    transform = transforms.Compose([transforms.Resize(size=shape_img)])
    dst = ImageDataset(
            images_all, np.asarray(labels_all, dtype=int),
            transform=transform)
    return dst


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = \
        ','.join(str(gpu) for gpu in args.visible_gpus)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = args.dataset
    root_path = '.'
    data_path = os.path.join(root_path, 'data')
    save_path = os.path.join(root_path, 'results/iDLG_%s' % dataset)

    # lr = 1.0
    initial_lr = args.lr
    num_dummy = 1
    Iteration = args.max_iter
    num_exp = 1000

    tt = transforms.Compose([transforms.ToTensor()])
    tp = transforms.Compose([transforms.ToPILImage()])

    print(dataset, 'root_path:', root_path)
    print(dataset, 'data_path:', data_path)
    print(dataset, 'save_path:', save_path)

    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    """ load data """
    cmap = "viridis"
    if dataset == 'MNIST':
        cmap = "gray"
        shape_img = (28, 28)
        num_classes = 10
        channel = 1
        hidden = 588
        dst = datasets.MNIST(data_path, download=True)

    elif dataset == 'cifar100':
        shape_img = (32, 32)
        num_classes = 100
        channel = 3
        hidden = 768
        dst = datasets.CIFAR100(data_path, download=True)

    elif dataset == 'lfw':
        shape_img = (32, 32)
        num_classes = 5749
        channel = 3
        hidden = 768
        lfw_path = os.path.join(root_path, '../data/lfw')
        dst = lfw_dataset(lfw_path, shape_img)

    elif dataset == 'CheXpert':
        dst = CheXpertDataset(csv_path='./idlg_data_entry.csv')
    else:
        exit('unknown dataset')

    ''' train DLG and iDLG '''
    for idx_net in range(num_exp):
        net = LeNet(channel=channel, hideen=hidden, num_classes=num_classes)
        net.apply(weights_init)

        print('running %d|%d experiment' % (idx_net, num_exp))
        net = net.to(device)
        np.random.seed(idx_net)
        idx_shuffle = np.random.permutation(len(dst))

        for method in ['DLG', 'iDLG']:
            print('%s, Try to generate %d images' % (method, num_dummy))

            criterion = nn.CrossEntropyLoss().to(device)
            imidx_list = []

            for imidx in range(num_dummy):
                idx = idx_shuffle[imidx]
                imidx_list.append(idx)
                tmp_datum = tt(dst[idx][0]).float().to(device)
                tmp_datum = tmp_datum.view(1, *tmp_datum.size())
                tmp_label = torch.Tensor([dst[idx][1]]).long().to(device)
                tmp_label = tmp_label.view(1, )
                if imidx == 0:
                    gt_data = tmp_datum
                    gt_label = tmp_label
                else:
                    gt_data = torch.cat((gt_data, tmp_datum), dim=0)
                    gt_label = torch.cat((gt_label, tmp_label), dim=0)

            # compute original gradient
            out = net(gt_data)
            y = criterion(out, gt_label)
            dy_dx = torch.autograd.grad(y, net.parameters())
            original_dy_dx = list((t.detach().clone() for t in dy_dx))

            # generate dummy data and label
            dummy_data = torch.randn(gt_data.size()).to(
                device).requires_grad_(True)
            dummy_label = torch.randn(
                (gt_data.shape[0], num_classes)
                ).to(device).requires_grad_(True)

            # truncated dummy image and label
            dummy_data.data = torch.clamp(dummy_data + 0.5, 0, 1)

            if method == 'DLG':
                # optim_obj = [dummy_data, dummy_label]
                optimizer = torch.optim.LBFGS(
                    [{'params': [dummy_data, dummy_label],
                      'initial_lr': initial_lr}],
                    lr=initial_lr
                    )
            elif method == 'iDLG':
                # optim_obj = [dummy_data, ]
                optimizer = torch.optim.LBFGS(
                    [{'params': [dummy_data, ], 'initial_lr': initial_lr}],
                    lr=initial_lr)
                # predict the ground-truth label
                label_pred = torch.argmin(
                    torch.sum(original_dy_dx[-2], dim=-1),
                    dim=-1).detach().reshape((1,)).requires_grad_(False)

            history = []
            history_iters = []
            losses = []
            mses = []
            train_iters = []

            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.95, last_epoch=0.1)
            print('lr =', initial_lr)
            for iters in range(Iteration):

                def closure():
                    optimizer.zero_grad()
                    pred = net(dummy_data)
                    if method == 'DLG':
                        dummy_loss = - torch.mean(
                            torch.sum(
                                torch.softmax(dummy_label, -1) *
                                torch.log(torch.softmax(pred, -1)),
                                dim=-1)
                        )
                        # dummy_loss = criterion(pred, gt_label)
                    elif method == 'iDLG':
                        dummy_loss = criterion(pred, label_pred)

                    dummy_dy_dx = torch.autograd.grad(
                        dummy_loss, net.parameters(), create_graph=True
                        )

                    grad_diff = 0
                    for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                        grad_diff += ((gx - gy) ** 2).sum()
                    grad_diff.backward()
                    # nn.utils.clip_grad_norm_([dummy_data], max_norm=0.1)
                    return grad_diff

                optimizer.step(closure)

                # pixel value clip
                # dummy_data.data.clamp(0, 1)
                dummy_data.data = torch.clamp(dummy_data, 0, 1)
                # dummy_gg = dummy_data.grad.data.cpu().numpy()
                # print("Dummy data gradient max: %f, min: %f" %
                #       (np.max(dummy_gg), np.min(dummy_gg)))
                # dummy_label.data = torch.clamp(dummy_label, 0, 1)
                # print(dummy_data.data.min(), dummy_data.data.max())

                current_loss = closure().item()
                train_iters.append(iters)
                losses.append(current_loss)
                mses.append(torch.mean((dummy_data-gt_data)**2).item())
                scheduler.step()

                # if iters % int(Iteration / 30) == 0:
                if iters % 30 == 0:
                    current_time = str(time.strftime(
                        "[%Y-%m-%d %H:%M:%S]", time.localtime()))
                    print(current_time, iters,
                          'loss = %.8f, mse = %.8f' % (current_loss, mses[-1]))
                    history.append([
                        tp(dummy_data[imidx].cpu())
                        for imidx in range(num_dummy)
                        ])
                    history_iters.append(iters)

                    for imidx in range(num_dummy):
                        plt.figure(figsize=(12, 8))
                        plt.subplot(3, 10, 1)
                        plt.imshow(tp(gt_data[imidx].cpu()), cmap=cmap)
                        for i in range(min(len(history), 29)):
                            plt.subplot(3, 10, i + 2)
                            plt.imshow(history[i][imidx], cmap=cmap)
                            plt.title('iter=%d' % (history_iters[i]))
                            plt.axis('off')
                        if method == 'DLG':
                            plt.savefig(
                                '%s/DLG_on_%s_%05d.png' %
                                (save_path, imidx_list, imidx_list[imidx])
                                )
                            plt.close()
                        elif method == 'iDLG':
                            plt.savefig(
                                '%s/iDLG_on_%s_%05d.png' %
                                (save_path, imidx_list, imidx_list[imidx])
                                )
                            plt.close()

                    # if current_loss < 0.000001:  # converge
                    if mses[-1] < 1e-6:
                        break

            if method == 'DLG':
                loss_DLG = losses
                label_DLG = torch.argmax(dummy_label, dim=-1).detach().item()
                mse_DLG = mses
            elif method == 'iDLG':
                loss_iDLG = losses
                label_iDLG = label_pred.item()
                mse_iDLG = mses

        print('imidx_list:', imidx_list)
        print('loss_DLG:', loss_DLG[-1], 'loss_iDLG:', loss_iDLG[-1])
        print('mse_DLG:', mse_DLG[-1], 'mse_iDLG:', mse_iDLG[-1])
        print('gt_label:', gt_label.detach().cpu().data.numpy(),
              'lab_DLG:', label_DLG, 'lab_iDLG:', label_iDLG)

        print('----------------------\n\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="iDLG"
        )

    parser.add_argument(
            "--visible_gpus",
            type=int,
            nargs='+',
            default=[0],
            help="CUDA visible gpus")

    parser.add_argument(
            "--lr",
            type=float,
            default=0.5,
            help="learning rate")

    parser.add_argument(
            "--max_iter",
            type=int,
            default=300,
            help="maximum iterations")

    parser.add_argument(
            "--dataset",
            type=str,
            default="MNIST",
            help="use image dataset",
            choices=["MNIST", "cifar100"])

    args = parser.parse_args()

    main(args)
