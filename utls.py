import time
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _closure(optimizer, dummy_data, dummy_label, label_pred, method, criterion,
             net, original_dy_dx):
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
    for dg, og in zip(dummy_dy_dx, original_dy_dx):
        grad_diff += ((dg - og)**2).sum()
    grad_diff.backward()
    # nn.utils.clip_grad_norm_([dummy_data], max_norm=0.1)
    return grad_diff


def get_current_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))


def plot_dummy_x(imidx, cmap, tp, gt_data, history, history_iters, save_path,
                 method, selected_indices, idx_exp):
    plt.figure(figsize=(12, 6))
    plt.subplot(3, 10, 1)
    plt.imshow(tp(gt_data[imidx].cpu()), cmap=cmap)
    for i in range(min(len(history), 29)):
        plt.subplot(3, 10, i + 2)
        plt.imshow(history[i][imidx], cmap=cmap)
        plt.title('iter=%d' % (history_iters[i]))
        plt.axis('off')

    # save plot figure
    plt.savefig('%s/%s_on_%05d.png' %
                (save_path, method, selected_indices[idx_exp]),
                dpi=200,
                bbox_inches="tight")
    plt.close()


def plot_mse_curve(mses, iters, save_path, method, selected_indices, idx_exp):
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


## Min-Max normalization
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
