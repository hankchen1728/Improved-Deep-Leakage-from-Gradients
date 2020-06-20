import os
import argparse
import numpy as np

import torch
import torch.nn.functional as F

import torchvision
from torchvision import datasets, transforms

# Import customize module
import models
import utils

print('torch version %s and torchvision version %s' % (torch.__version__,
                                                       torchvision.__version__))


def dlg_method(dummy_data, dummy_label, original_dy_dx,
               net, criterion, optimizer, tt, max_iters=300):
    dummy_grads = []
    dummy_lbfgs_num_iter = []
    history = []

    for iters in range(max_iters):

        closure_grads = []
        def closure():
            optimizer.zero_grad()

            pred = net(dummy_data)
            dummy_onehot_label = F.softmax(dummy_label, dim=-1)
            # TODO: fix the gt_label to dummy_label in both code and slides.
            dummy_loss = criterion(pred, dummy_onehot_label)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(),
                                              create_graph=True)

            grad_diff = 0
            grad_count = 0

            for dg, og in zip(dummy_dy_dx, original_dy_dx):
                grad_diff += ((dg - og) ** 2).sum()
                grad_count += dg.nelement()
            # grad_diff = grad_diff / grad_count * 1000

            closure_grads.append(dummy_dy_dx)
            grad_diff.backward()
            return grad_diff

        optimizer.step(closure)
        dummy_grads.append(closure_grads[-1])
        dummy_lbfgs_num_iter.append(len(closure_grads))
        history.append([dummy_data[0].cpu(),
                        torch.argmax(dummy_label, dim=-1).item()])

        if iters % 10 == 0:
            current_loss = closure()
            print('%3d %2.6f' % (iters, current_loss.item()))
            if current_loss.item() < 1e-5:
                break

    return dummy_grads, dummy_lbfgs_num_iter, history


def main(args):
    # Set GPUs and device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.visible_gpus
    device = 'cuda' if torch.cuda.is_available else 'cpu'
    print('Running on %s' % device)

    # Set environment
    torch.manual_seed(args.seed)

    # Get dataset and define transformations
    dst = datasets.CIFAR100(args.data_dir, download=True)
    tp = transforms.Compose([transforms.Resize(32),
                             transforms.CenterCrop(32),
                             transforms.ToTensor()])
    tt = transforms.ToPILImage()

    # Construct model and intiaize weights
    net = models.LeNet().to(device)
    net.apply(utils.weights_init)

    # Define criterion
    criterion = utils.cross_entropy_for_onehot

    # Get attack data and label
    gt_data = tp(dst[args.image_idx][0]).to(device)
    gt_data = gt_data.view(1, *gt_data.size())
    gt_image = tt(gt_data[0].cpu())
    gt_label = torch.Tensor([dst[args.image_idx][1]]).long().to(device)
    gt_label = gt_label.view(1,)
    gt_onehot_label = utils.label_to_onehot(gt_label, num_classes=100)

    # Compute original gradient
    out = net(gt_data)
    y = criterion(out, gt_onehot_label)
    dy_dx = torch.autograd.grad(y, net.parameters())

    # Share the gradients with other clients
    original_dy_dx = list((_.detach().clone() for _ in dy_dx))

    # Generate dummy data and label
    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
    dummy_label = torch.randn(
        gt_onehot_label.size()).to(device).requires_grad_(True)

    dummy_init_image = tt(dummy_data[0].cpu())
    dummy_init_label = torch.argmax(dummy_label, dim=-1)

    # Define optimizer
    optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

    # Run DLG method
    dummy_grads, dummy_lbfgs_num_iter, history = \
        dlg_method(dummy_data, dummy_label, original_dy_dx,
                   net, criterion, optimizer, tt, max_iters=args.max_iters)

    # Save model
    params_path = os.path.join(args.exp_dir, '%04d_params.pkl' % args.image_idx)
    torch.save(net.state_dict(), params_path)
    print('Save model parameters to %s' % params_path)

    # Index computation functions
    compute_l2norm = lambda x: (x ** 2).sum().item() ** 0.5
    compute_min = lambda x: x.min().item()
    compute_max = lambda x: x.max().item()
    compute_mean = lambda x: x.mean().item()
    compute_median = lambda x: x.median().item()

    original_grads_norm = [compute_l2norm(e) for e in original_dy_dx]
    original_grads_min = [compute_min(e) for e in original_dy_dx]
    original_grads_max = [compute_max(e) for e in original_dy_dx]
    original_grads_mean = [compute_mean(e) for e in original_dy_dx]
    original_grads_median = [compute_median(e) for e in original_dy_dx]

    dummy_grads_norm = np.array(
        [[compute_l2norm(e) for e in r] for r in dummy_grads])
    dummy_grads_min = np.array(
        [[compute_min(e) for e in r] for r in dummy_grads])
    dummy_grads_max = np.array(
        [[compute_max(e) for e in r] for r in dummy_grads])
    dummy_grads_mean = np.array(
        [[compute_mean(e) for e in r] for r in dummy_grads])
    dummy_grads_median = np.array(
        [[compute_median(e) for e in r] for r in dummy_grads])

    # Plot and save figures
    fig_dir = os.path.join(args.exp_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    img_history = [[tt(hd), hl] for hd, hl in history]
    utils.plot_history(
        [gt_image, gt_label], [dummy_init_image, dummy_init_label], img_history,
        title='Image %04d History' % args.image_idx,
        fig_path=os.path.join(fig_dir, '%04d_history.png' % args.image_idx))

    utils.plot_convergency_curve(
        dummy_grads_norm, original_grads_norm,
        title='Image %04d L2 Norm Convergency' % args.image_idx,
        fig_path=os.path.join(fig_dir, '%04d_l2norm.png' % args.image_idx))

    utils.plot_convergency_curve(
        dummy_grads_min, original_grads_min,
        title='Image %04d Min Value Convergency' % args.image_idx,
        fig_path=os.path.join(fig_dir, '%04d_min.png' % args.image_idx))

    utils.plot_convergency_curve(
        dummy_grads_max, original_grads_max,
        title='Image %04d Max Value Convergency' % args.image_idx,
        fig_path=os.path.join(fig_dir, '%04d_max.png' % args.image_idx))

    utils.plot_convergency_curve(
        dummy_grads_mean, original_grads_mean,
        title='Image %04d Mean Convergency' % args.image_idx,
        fig_path=os.path.join(fig_dir, '%04d_mean.png' % args.image_idx))

    utils.plot_convergency_curve(
        dummy_grads_median, original_grads_median,
        title='Image %04d Median Value Convergency' % args.image_idx,
        fig_path=os.path.join(fig_dir, '%04d_median.png' % args.image_idx))

    compute_mse = lambda x, y: ((x - y) ** 2).sum().item() ** 0.5
    final_mse = compute_mse(dummy_data, gt_data)
    converged = final_mse < args.mse_tol
    print('Converged!! (MSE=%2.6f)' % final_mse
          if converged else 'Diverged!! (MSE=%2.6f)' % final_mse)

    # Save MSEs
    mses = np.array([compute_mse(hd.cuda(), gt_data) for hd, _ in history])
    with open(os.path.join(args.exp_dir,
                           '%04d_mses.npy' % args.image_idx), 'wb') as opf:
        np.save(opf, mses)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters Analysis')

    # Environment settings
    parser.add_argument('-vg', '--visible-gpus', dest='visible_gpus',
                        default='6', help='set cuda visible devices')
    parser.add_argument('--seed', dest='seed', default=30, type=int,
                        metavar='S', help='random seed (default: 30)')

    # Paths settings
    parser.add_argument('--data-dir', dest='data_dir', default='../../data',
                        metavar='PATH', help='data directory path')
    parser.add_argument('--exp-dir', dest='exp_dir', default='../../exp',
                        metavar='PATH', help='experiments home directory path')
    parser.add_argument('-n', '--exp-name', dest='exp_name',
                        help='experiment name')

    # Training settings
    parser.add_argument('--image-idx', dest='image_idx', default=10, type=int,
                        metavar='S', help='image index (default: 10)')
    parser.add_argument('--max-iters', dest='max_iters', default=300, type=int,
                        metavar='S', help='max iterations (default: 300)')
    parser.add_argument('--mse-tol', dest='mse_tol', default=3.0, type=float,
                        metavar='S', help='MSE tolerance (default: 3.0)')

    # Version
    parser.add_argument('--version', action='version', version='%(prog)s 0.1.0')
    args = parser.parse_args()

    # Set experiment directory
    args.exp_dir = os.path.join(args.exp_dir, args.exp_name)
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)

    print('Start experiment under %s with following settings:' % args.exp_dir)
    print('IMAGE_IDX=%d, SEED=%d, MSE_TOL=%.6f' % (
        args.image_idx,args.seed, args.mse_tol))
    main(args)

