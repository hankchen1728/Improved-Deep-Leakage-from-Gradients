import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def label_to_onehot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes,
    							device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target


def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))


def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)


def plot_history(gt, di, history, title, cols=6, log_interval=10, fig_path=None):
    rows = (len(history) // log_interval + 2) // cols + 1
    fig, axes = plt.subplots(rows, cols, figsize=(15, 2.5*rows+1.5),
    						 constrained_layout=True)
    fig.suptitle(title, fontsize=18)

    # Plot ground truth and dummy initial image and label, respectively
    axes[0, 0].imshow(gt[0])
    axes[0, 0].set_title('GT Image with Label %d' % gt[1].item())
    axes[0, 0].axis('off')

    axes[0, 1].imshow(di[0])
    axes[0, 1].set_title('Dummy Init with Label %d' % di[1].item())
    axes[0, 1].axis('off')

    for i in range(0, len(history), log_interval):
        i_log = i // log_interval + 2
        axes[i_log // cols, i_log % cols].imshow(history[i][0])
        axes[i_log // cols, i_log % cols].set_title(
            'Iter {:<3d} with label {:d}'.format(i, history[i][1]))
        axes[i_log // cols, i_log % cols].axis('off')

    # Delete empty axes
    for i in range(i_log%cols+1, cols):
        fig.delaxes(axes[i_log // cols, i])

    # Save figure
    if fig_path is not None:
        fig.savefig(fig_path, bbox_inches='tight')
    fig.show()


def plot_convergency_curve(grads_index, original_index, title, fig_path=None):
    fig, axes = plt.subplots(2, 5, figsize=(16, 9), constrained_layout=True)
    fig.suptitle(title, fontsize=18)

    for i in range(10):
        axes[i // 5, i % 5].plot(grads_index[:, i], color='b')
        axes[i // 5, i % 5].axhline(original_index[i], color='r')

    # Save figure
    if fig_path is not None:
        fig.savefig(fig_path, bbox_inches='tight')
    fig.show()

