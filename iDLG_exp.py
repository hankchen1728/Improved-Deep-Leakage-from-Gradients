import os, argparse, functools
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from models import ConvNet, config_net, config_resnet18
from dataset import CheXpertDataset, ImageDataset, lfw_dataset, DatasetParams
from utils import _closure, get_current_time, plot_dummy_x, plot_mse_curve


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

    data_params = DatasetParams()
    data_params.config(name=dataset, root_path=root_path, data_path=data_path)

    shape_img = data_params.shape_img
    num_classes = data_params.num_classes
    channel = data_params.channel
    dst = data_params.dst
    selected_indices = data_params.selected_indices
    cmap = data_params.cmap

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
            orig_dy_dx = list((t.detach().clone() for t in dy_dx))
            if args.grad_norm:
                grad_max = [x.max().item() for x in orig_dy_dx]
                grad_min = [x.min().item() for x in orig_dy_dx]
                orig_dy_dx = [
                    (g - g_min) / (g_max - g_min)
                    for g, g_min, g_max in zip(orig_dy_dx, grad_min, grad_max)
                ]

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
                    lr=initial_lr,
                    max_iter=50,
                    tolerance_grad=1e-9,
                    tolerance_change=1e-11,
                    history_size=250,
                    line_search_fn="strong_wolfe")
            elif method == 'iDLG':
                # optim_obj = [dummy_data, ]
                optimizer = torch.optim.LBFGS(
                    [{
                        'params': [dummy_data],
                        'initial_lr': initial_lr
                    }],
                    lr=initial_lr,
                    max_iter=50,
                    tolerance_grad=1e-9,
                    tolerance_change=1e-11,
                    history_size=250,
                    line_search_fn="strong_wolfe")
                # predict the ground-truth label
                label_pred = torch.argmin(torch.sum(orig_dy_dx[-2], dim=-1),
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

                closure = functools.partial(_closure,
                                            optimizer=optimizer,
                                            dummy_data=dummy_data,
                                            dummy_label=dummy_label,
                                            label_pred=label_pred,
                                            method=method,
                                            criterion=criterion,
                                            net=net,
                                            orig_dy_dx=orig_dy_dx,
                                            grad_norm=args.grad_norm)

                optimizer.step(closure)

                # pixel value clamp
                if args.add_clamp:
                    dummy_data.data = torch.clamp(dummy_data, 0, 1)

                current_loss = closure().item()
                train_iters.append(iters)
                losses.append(current_loss)
                mses.append(torch.mean((dummy_data - gt_data)**2).item())
                scheduler.step()

                if iters % plot_steps == 0:
                    current_time = get_current_time()
                    print(current_time, iters,
                          'loss = %.8f, mse = %.8f' % (current_loss, mses[-1]))
                    history.append([
                        tp(dummy_data[imidx].cpu())
                        for imidx in range(num_dummy)
                    ])
                    history_iters.append(iters)

                    for imidx in range(num_dummy):
                        plot_dummy_x(imidx, cmap, tp, gt_data, history,
                                     history_iters, save_path, method,
                                     selected_indices, idx_exp)

                    # if current_loss < 0.000001:
                    # converge
                    if mses[-1] < 1e-4:
                        break
            if mses[-1] < 1e-3:
                num_success += 1
            # Save mse curve
            plot_mse_curve(mses, iters, save_path, method, selected_indices,
                           idx_exp)

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

    parser.add_argument("--grad_norm", action="store_true")

    args = parser.parse_args()

    main(args)
