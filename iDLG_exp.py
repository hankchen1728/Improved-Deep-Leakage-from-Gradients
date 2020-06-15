import time
import os
import argparse
import PIL.Image as Image

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from dataset import CheXpertDataset
from models import ConvNet


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


def config_net(net_name="", input_shape=(3, 32, 32), num_classes=10):
    assert net_name in ["CNN_L2D1", "CNN_L2D2", "CNN_L4D1", "CNN_L4D2"]
    if net_name == "CNN_L2D1":
        conv_channels = [32, 64]
    elif net_name == "CNN_L2D2":
        conv_channels = [64, 128]
    elif net_name == "CNN_L4D1":
        conv_channels = [32, 64, 128, 256]
    elif net_name == "CNN_L4D2":
        conv_channels = [64, 128, 256, 512]
    net = ConvNet(
        image_shape=input_shape,
        conv_channels=conv_channels,
        num_classes=num_classes
    )
    return net


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = \
        ','.join(str(gpu) for gpu in args.visible_gpus)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = args.dataset
    net_name = args.cnn_name
    root_path = '.'
    data_path = os.path.join(root_path, "data")
    save_path = os.path.join(
        root_path, "results", "iDLG_%s_%s" % (dataset, net_name)
    )
    if args.add_clamp:
        save_path += "_clamp"

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
        dst = datasets.MNIST(data_path, download=True)

    elif dataset == 'cifar100':
        shape_img = (32, 32)
        num_classes = 100
        channel = 3
        dst = datasets.CIFAR100(data_path, download=True)

    elif dataset == 'lfw':
        shape_img = (32, 32)
        num_classes = 5749
        channel = 3
        lfw_path = os.path.join(root_path, '../data/lfw')
        dst = lfw_dataset(lfw_path, shape_img)

    elif dataset == 'CheXpert':
        shape_img = (224, 224)
        num_classes = 2
        channel = 3
        dst = CheXpertDataset(csv_path='./idlg_data_entry.csv')
    else:
        exit('unknown dataset')

    # Build ConvNet
    net = config_net(
        net_name=net_name, input_shape=(channel,)+shape_img,
        num_classes=num_classes
    )
    net = net.to(device)

    # Load model pretrain weights
    if os.path.isfile(args.model_ckpt):
        ckpt = torch.load(args.model_ckpt)
        net.load_state_dict(ckpt)

    criterion = nn.CrossEntropyLoss().to(device)
    ''' train DLG and iDLG '''
    for idx_exp in range(num_exp):

        print('running %d|%d experiment' % (idx_exp, num_exp))
        np.random.seed(idx_exp)
        idx_shuffle = np.random.permutation(len(dst))

        for method in ['DLG', 'iDLG']:
            print('%s, Try to generate %d images' % (method, num_dummy))

            # criterion = nn.CrossEntropyLoss().to(device)
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
            print(gt_data.shape)
            out = net(gt_data)
            y = criterion(out, gt_label)
            dy_dx = torch.autograd.grad(y, net.parameters())
            original_dy_dx = list((t.detach().clone() for t in dy_dx))

            # generate dummy data and label
            torch.cuda.manual_seed_all(10)
            dummy_data = torch.randn(gt_data.size()).to(
                device).requires_grad_(True)
            dummy_label = torch.randn(
                (gt_data.shape[0], num_classes)
                ).to(device).requires_grad_(True)

            # truncated dummy image and label
            dummy_data.data = torch.clamp(dummy_data + 0.5, 0, 1)
            dummy_label.data = torch.clamp(dummy_label + 0.5, 0, 1)

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
                if args.add_clamp:
                    dummy_data.data = torch.clamp(dummy_data, 0, 1)

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
            "--cnn_name",
            type=str,
            default="CNN_L2D1",
            choices=["CNN_L2D1", "CNN_L2D2", "CNN_L4D1", "CNN_L4D2"],
            help="CNN config")

    parser.add_argument(
            "--model_ckpt",
            type=str,
            default="",
            help="Model checkpoint")

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
            choices=["MNIST", "cifar100", "CheXpert"])

    parser.add_argument(
            "--add_clamp",
            action="store_true")

    args = parser.parse_args()

    main(args)
