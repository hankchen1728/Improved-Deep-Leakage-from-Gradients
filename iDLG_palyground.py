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
from models import ConvNet, config_net, config_resnet18
from dataset import CheXpertDataset, ImageDataset, lfw_dataset


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = \
        ','.join(str(gpu) for gpu in args.visible_gpus)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = args.dataset
    net_name = args.cnn_name
    root_path = '.'
    data_path = os.path.join(root_path, "data")
    save_path = os.path.join(root_path, "results",
                             "iDLG_%s_%s" % (dataset, net_name))

    if args.add_clamp:
        save_path += "_clamp"

    # Some running setting
    initial_lr = args.lr
    num_dummy = 1
    Iteration = args.max_iter
    plot_steps = args.plot_steps
    # run_methods = ["iDLG", "DLG"]
    run_methods = ["iDLG"]

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
    selected_indices = []
    cmap = "viridis"
    if dataset == 'MNIST':
        shape_img = (28, 28)
        num_classes = 10
        channel = 1
        # hidden = 588
        dst = datasets.MNIST(data_path, download=True)
        selected_indices = np.array([
            1,
            21,
            34,  # label : 0
            3,
            6,
            8,  # label : 1
            5,
            16,
            25,  # label : 2
            7,
            10,
            12,  # label : 3
            2,
            9,
            20,  # label : 4
            0,
            11,
            35,  # label : 5
            13,
            18,
            32,  # label : 6
            15,
            29,
            38,  # label : 7
            17,
            31,
            41,  # label : 8
            4,
            19,
            22  # label : 9
        ])
    elif dataset == "cifar10":
        shape_img = (32, 32)
        num_classes = 100
        channel = 3
        # hidden = 768
        dst = datasets.CIFAR10(data_path, download=True)
        selected_indices = np.array([
            29, 30, 35, 49, 77, 93, 115, 116, 129, 165, 4, 5, 32, 44, 45, 46,
            60, 61, 64, 65, 6, 13, 18, 24, 41, 42, 47, 48, 54, 55, 9, 17, 21,
            26, 33, 36, 38, 39, 59, 74, 3, 10, 20, 28, 34, 58, 66, 82, 86, 89,
            27, 40, 51, 56, 70, 81, 83, 107, 128, 148, 0, 19, 22, 23, 25, 72,
            95, 103, 104, 117, 7, 11, 12, 37, 43, 52, 68, 73, 84, 85, 8, 62,
            69, 92, 100, 106, 111, 135, 139, 155, 1, 2, 14, 15, 16, 31, 50, 53,
            67, 71
        ])
        num_exp = selected_indices.shape[0]
    elif dataset == 'cifar100':
        shape_img = (32, 32)
        num_classes = 100
        channel = 3
        # hidden = 768
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
        channel = 1
        resize_t = transforms.Resize(shape_img)
        dst = CheXpertDataset(csv_path='./idlg_data_entry.csv',
                              channels=channel,
                              transforms=resize_t)
        selected_indices = np.arange(0, 30, 3)
    else:
        exit('unknown dataset')

    if channel == 1:
        cmap = "gray"

    # Build ConvNet
    net = config_net(net_name=net_name,
                     input_shape=(channel, ) + shape_img,
                     num_classes=num_classes)
    net = net.to(device)
    # net.eval()

    # Load model pretrain weights
    if os.path.isfile(args.model_ckpt):
        ckpt = torch.load(args.model_ckpt)
        net.load_state_dict(ckpt)

    num_success = 0
    num_exp = len(selected_indices)
    criterion = nn.CrossEntropyLoss().to(device)
    ''' train DLG and iDLG '''
    for idx_exp in range(num_exp):

        print('running %d|%d experiment' % (idx_exp, num_exp))
        np.random.seed(idx_exp)
        # idx_shuffle = np.random.permutation(len(dst))

        for method in run_methods:
            print('%s, Try to generate %d images' % (method, num_dummy))

            # criterion = nn.CrossEntropyLoss().to(device)
            imidx_list = []

            # get ground truth image and label
            idx = selected_indices[idx_exp]
            imidx_list.append(idx)
            tmp_datum = tt(dst[idx][0]).float().to(device)
            tmp_datum = tmp_datum.view(1, *tmp_datum.size())
            tmp_label = torch.Tensor([dst[idx][1]]).long().to(device)
            tmp_label = tmp_label.view(1, )
            gt_data = tmp_datum
            gt_label = tmp_label

            # compute original gradient
            out = net(gt_data)
            y = criterion(out, gt_label)
            dy_dx = torch.autograd.grad(y, net.parameters())
            original_dy_dx = list((t.detach().clone() for t in dy_dx))

            # generate dummy data and label
            torch.manual_seed(10)
            dummy_data = torch.randn(
                gt_data.size()).to(device).requires_grad_(True)
            dummy_label = torch.randn(
                (gt_data.shape[0],
                 num_classes)).to(device).requires_grad_(True)

            # truncated dummy image and label
            if args.add_clamp:
                dummy_data.data = torch.clamp(dummy_data + 0.5, 0, 1)
                dummy_label.data = torch.clamp(dummy_label + 0.5, 0, 1)

            if method == 'DLG':
                # optim_obj = [dummy_data, dummy_label]
                optimizer = torch.optim.LBFGS(
                    [{
                        'params': [dummy_data, dummy_label],
                        'initial_lr': initial_lr
                    }],
                    lr=initial_lr)
            elif method == 'iDLG':
                # optim_obj = [dummy_data, ]
                optimizer = torch.optim.LBFGS([{
                    'params': [
                        dummy_data,
                    ],
                    'initial_lr': initial_lr
                }],
                                              lr=initial_lr)
                # predict the ground-truth label
                label_pred = torch.argmin(torch.sum(original_dy_dx[-2],
                                                    dim=-1),
                                          dim=-1).detach().reshape(
                                              (1, )).requires_grad_(False)

            history = []
            history_iters = []
            losses = []
            mses = []
            train_iters = []

            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                        step_size=30,
                                                        gamma=0.95,
                                                        last_epoch=-1)
            print('lr =', initial_lr)
            for iters in range(Iteration):

                def closure():
                    optimizer.zero_grad()
                    pred = net(dummy_data)
                    if method == 'DLG':
                        dummy_loss = -torch.mean(
                            torch.sum(torch.softmax(dummy_label, -1) *
                                      torch.log(torch.softmax(pred, -1)),
                                      dim=-1))
                        # dummy_loss = criterion(pred, gt_label)
                    elif method == 'iDLG':
                        dummy_loss = criterion(pred, label_pred)

                    dummy_dy_dx = torch.autograd.grad(dummy_loss,
                                                      net.parameters(),
                                                      create_graph=True)

                    grad_diff = 0
                    for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                        grad_diff += ((gx - gy)**2).sum()
                    grad_diff.backward()
                    # nn.utils.clip_grad_norm_([dummy_data], max_norm=0.1)
                    return grad_diff

                optimizer.step(closure)

                # pixel value clip
                if args.add_clamp:
                    dummy_data.data = torch.clamp(dummy_data, 0, 1)
                # print(dummy_data.data.max(), dummy_data.data.min())
                # if args.add_clamp:
                #     # Min-max normalization
                #     dummy_max = dummy_data.data.max()
                #     dummy_min = dummy_data.data.min()
                #     dummy_diff = dummy_max - dummy_min
                #     if dummy_diff != 0:
                #         if dummy_max > 1 and dummy_min > 0:
                #             dummy_data.data = \
                #                 (dummy_data - dummy_min) / dummy_diff \
                #                 * (1 - dummy_min) + dummy_min
                #         elif dummy_max < 1 and dummy_min < 0:
                #             dummy_data.data = \
                #                 (dummy_data - dummy_min) / dummy_diff \
                #                 * dummy_max
                #         elif dummy_max > 1 and dummy_min < 0:
                #             dummy_data.data = \
                #                 (dummy_data - dummy_min) / dummy_diff
                #         print(dummy_data.data.max(), dummy_data.data.min())

                current_loss = closure().item()
                train_iters.append(iters)
                losses.append(current_loss)
                mses.append(torch.mean((dummy_data - gt_data)**2).item())
                scheduler.step()

                if iters % plot_steps == 0:
                    current_time = str(
                        time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
                    print(current_time, iters,
                          'loss = %.8f, mse = %.8f' % (current_loss, mses[-1]))
                    history.append([
                        tp(dummy_data[imidx].cpu())
                        for imidx in range(num_dummy)
                    ])
                    history_iters.append(iters)

                    for imidx in range(num_dummy):
                        plt.figure(figsize=(12, 6))
                        plt.subplot(3, 10, 1)
                        plt.imshow(tp(gt_data[imidx].cpu()), cmap=cmap)
                        for i in range(min(len(history), 29)):
                            plt.subplot(3, 10, i + 2)
                            plt.imshow(history[i][imidx], cmap=cmap)
                            plt.title('iter=%d' % (history_iters[i]))
                            plt.axis('off')

                        # save plot figure
                        plt.savefig(
                            '%s/%s_on_%05d.png' %
                            (save_path, method, selected_indices[idx_exp]),
                            dpi=200,
                            bbox_inches="tight")
                        plt.close()

                    # if current_loss < 0.000001:
                    # converge
                    if mses[-1] < 1e-6:
                        break
            if mses[-1] < 1e-3:
                num_success += 1
            # Save mse curve
            plt.figure(figsize=(6, 4))
            plt.plot(mses)
            plt.xlabel("Iterations")
            plt.ylabel("MSE")
            plt.ylim(-1e-3, 0.1)
            plt.xlim(-1, (iters // 100 + 1) * 100)
            plt.title("Reconstruction MSE Loss")
            plt.savefig("%s/mse_%s_on_%05d.png" %
                        (save_path, method, selected_indices[idx_exp]))
            plt.close()

            if method == 'DLG':
                loss_DLG = losses
                label_DLG = torch.argmax(dummy_label, dim=-1).detach().item()
                mse_DLG = mses
            elif method == 'iDLG':
                loss_iDLG = losses
                label_iDLG = label_pred.item()
                mse_iDLG = mses

        print('gt_label:', gt_label.detach().cpu().data.numpy())
        if "DLG" in run_methods:
            print('loss_DLG:', loss_DLG[-1], 'mse_DLG:', mse_DLG[-1],
                  'lab_DLG:', label_DLG)
        if "iDLG" in run_methods:
            print('loss_iDLG:', loss_iDLG[-1], 'mse_iDLG:', mse_iDLG[-1],
                  'lab_iDLG:', label_iDLG)

        print('----------------------\n\n')

    print("Number of successful recover:", num_success)

    # end


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="iDLG")

    parser.add_argument("--visible_gpus",
                        type=int,
                        nargs='+',
                        default=[0],
                        help="CUDA visible gpus")

    parser.add_argument("--cnn_name",
                        type=str,
                        default="CNN_L2D1",
                        choices=[
                            "CNN_L2D1", "CNN_L2D2", "CNN_L4D1", "CNN_L4D2",
                            "CNN_L4D4", "CNN_L6D2", "CNN_L7D2", "ResNet18"
                        ],
                        help="CNN config")

    parser.add_argument("--model_ckpt",
                        type=str,
                        default="",
                        help="Model checkpoint")

    parser.add_argument("--lr", type=float, default=0.5, help="learning rate")

    parser.add_argument("--max_iter",
                        type=int,
                        default=300,
                        help="maximum iterations")

    parser.add_argument("--plot_steps",
                        type=int,
                        default=5,
                        help="steps to plot the results")

    parser.add_argument("--dataset",
                        type=str,
                        default="MNIST",
                        help="use image dataset",
                        choices=["MNIST", "cifar10", "cifar100", "CheXpert"])

    parser.add_argument("--add_clamp", action="store_true")

    args = parser.parse_args()

    main(args)
