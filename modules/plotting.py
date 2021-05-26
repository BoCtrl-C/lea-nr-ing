import numpy as np
from matplotlib import pyplot as plt

import torch
from torch.nn import functional as F


CMAP = plt.get_cmap('tab10')
CMAP_OFFSET = - 1


def show_sequence(seq, pad=0, title=None, axis=False, figsize=None):
    """The function shows the provided sequence frames in a row.
    """

    frames = seq.shape[0]
    res = seq.shape[1]

    # build the sequence image
    img = np.ones((res + 2*pad, frames*(res + pad) + pad))
    for i, frame in enumerate(seq):
        img[pad:pad+res,pad+i*(res+pad):pad+i*(res+pad)+res] = frame

    plt.figure(figsize=figsize)
    plt.imshow(img, cmap='gray')
    if title is not None:
        plt.title(title)
    if not axis:
        plt.axis('off')
    plt.show()

def show_emb_sim(embs, labels=None, figsize=None):
    """The (i, j)-th entry of the displayed matrix represents the cosine
    similarity between embeddings i and j.
    """
    
    _, axs = plt.subplots(embs.shape[0], figsize=figsize)
    
    if labels is None: labels = range(embs.shape[0])
    
    s = torch.empty(embs.shape[0])
    for i, emb1 in enumerate(embs):
        for j, emb2 in enumerate(embs):
            s[j] = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
        axs[i].imshow(s.unsqueeze(0))
        axs[i].xaxis.set_visible(False)
        axs[i].set_yticks([0])
        axs[i].set_yticklabels([labels[i]])
    
    axs[-1].xaxis.set_visible(True)
    axs[-1].set_xticks(range(embs.shape[0]))
    axs[-1].set_xticklabels(labels)

    plt.show()

def show_pos_sim(encs, figsize=None):
    """Each subplot shows the cosine similarities between the represented
    positional encodings and the one highlighted in yellow. Positional encodings
    of the bottom-right pixels do not exist due to the autoregressive nature of
    the reconstruction task.
    """

    N = 4
    _, axs = plt.subplots(N, N, figsize=figsize)

    num_pos = encs.shape[0]
    res = int(np.sqrt(num_pos))
    s = torch.empty(num_pos)
    for i, y in enumerate(range(0, res, res//N)):
        for j, x in enumerate(range(0, res, res//N)):
            for k in range(num_pos):
                s[k] = F.cosine_similarity(
                    encs.view(res, res, - 1)[y, x].unsqueeze(0),
                    encs[k].unsqueeze(0)
                )
            s[-1] = -1 # SoS position
            axs[i, j].imshow(s.view(res, res))
            axs[i, j].xaxis.set_visible(False)
            axs[i, j].yaxis.set_visible(False)

    plt.show()

def show_reconstruction(gt, rec, axis=False, figsize=None):
    """The function displays the ground truth frame on the left and the
    reconstructed one on the right.
    """

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    ax1.imshow(gt, cmap='gray')
    ax1.set_title('Ground Truth')

    ax2.imshow(rec, cmap='gray')
    ax2.set_title('Reconstruction')
    
    if not axis:
        ax1.set_axis_off()
        ax2.set_axis_off()
    plt.show()

def show_number_line(
    A_2,
    nums=None,
    figsize=None
):
    """The function plots the provided first 2 PCs (A_2). If prj is True, only
    the first one is displayed.
    """

    plt.figure(figsize=figsize)
    
    # plot the arrow
    x_min = torch.min(A_2[:,0])
    x_max = torch.max(A_2[:,0])
    delta = x_max - x_min
    plt.arrow(
        x=x_min - 0.1*delta,
        y=0,
        dx=1.2*delta,
        dy=0,
        width=0.06,
        head_width=0.3,
        length_includes_head=True,
        fc='k'
    )
    
    if nums is None: nums = range(A_2.shape[0])
    
    for i, pcs in enumerate(A_2):
        plt.plot([pcs[0], pcs[0]], [pcs[1], 0], ls='--', c='k')
        plt.plot(pcs[0], pcs[1], marker='o', c=CMAP(nums[i] + CMAP_OFFSET))
        plt.annotate(
            nums[i],
            (pcs[0], 0),
            textcoords='offset points',
            xytext=(2, 3),
            c=CMAP(nums[i] + CMAP_OFFSET)
        )
    
    plt.title('The \"Number Line\"')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.axis('equal')

    plt.show()

def show_samples(batch, axis=False, figsize=None):
    """The function shows a square grid generated from the provided samples.
    Hence, the number of samples should be a perfect square.
    """

    n = int(np.sqrt(batch.shape[0]))
    
    _, axs = plt.subplots(n, n, figsize=figsize)
    for i, sample in enumerate(batch):
        ax = axs[i//n,i%n]
        ax.imshow(sample, cmap='gray')

        if not axis:
            ax.set_axis_off()
    
    plt.show()

def show_pmfs(
    pmfs,
    labels=None,
    num_xtrs=0,
    focus=None,
    alpha=0.5
):
    """The function displays the provided PMFs (input rows).
    """

    plt.figure()
    
    # curve colors
    tab20b = plt.get_cmap('tab20b')
    c_lrn = [CMAP(l + CMAP_OFFSET) for l in labels[:len(labels)-num_xtrs]]
    c_xtr = [tab20b(2 - i) for i in range(num_xtrs)]
    c = c_lrn + c_xtr

    for i, pmf in enumerate(pmfs):
        nonzeros = torch.nonzero(pmf)
        idxs = []
        for j in range(pmf.shape[0]):
            if (j + 1 in nonzeros or j in nonzeros or j - 1 in nonzeros)\
                and j < 11: # TODO: remove
                idxs.append(j)
        plt.plot(
            idxs,
            pmf[idxs],
            marker='o',
            ls='--',
            c=None if labels is None else c[i],
            alpha=1 if focus is None or labels[i] in focus else alpha,
            label=i if labels is None else labels[i]
        )

    plt.title('Distributions')
    plt.xlabel('Generated Numerosity')
    plt.ylabel('Relative Frequency')
    plt.xticks(range(torch.max(torch.nonzero(pmfs)[:,1]) + 2\
        - 1)) # TODO: remove
    plt.legend()
    plt.grid(axis='x',ls='--')
    
    plt.show()

def show_2d_hist(pmfs, from_one=False, figsize=None):
    """Each displayed column corresponds to a different generation histogram.
    """

    # cut off excess 0s
    last_nonzero = torch.max(torch.nonzero(pmfs)[:,1])
    if last_nonzero < pmfs.shape[0]:
        cut = pmfs.shape[0]
    else:
        cut = last_nonzero + 1
    pmfs = pmfs[:,:cut]
    
    # rotate the axes
    pmfs = torch.rot90(pmfs)
    
    plt.figure(figsize=figsize)

    im = plt.imshow(pmfs)
    
    # baseline
    if not from_one:
        plt.plot(
            [- 0.5, pmfs.shape[1] - 0.5],
            [pmfs.shape[0] - 0.5, pmfs.shape[0] - pmfs.shape[1] - 0.5],
            ls='--',
            c='r'
        )
    else:
        plt.plot(
            [- 0.5, pmfs.shape[1] - 0.5],
            [pmfs.shape[0] - 1.5, pmfs.shape[0] - pmfs.shape[1] - 1.5],
            ls='--',
            c='r'
        )

    cbar = plt.colorbar(im)
    plt.clim(0, 1)
    cbar.ax.set_ylabel('Relative Frequency', rotation=- 90, va='bottom')

    plt.title('2D Histogram')
    plt.xlabel('Seed')
    plt.ylabel('Generated Numerosity')

    # ticks
    ax = plt.gca()
    ax.set_xticks(range(pmfs.shape[1]))
    ax.set_xticklabels(range(
        0 if not from_one else 1,
        pmfs.shape[1] if not from_one else pmfs.shape[1] + 1
    ))
    ax.set_yticks(range(pmfs.shape[0]))
    ax.set_yticklabels(range(pmfs.shape[0] - 1, - 1, - 1))

    plt.show()