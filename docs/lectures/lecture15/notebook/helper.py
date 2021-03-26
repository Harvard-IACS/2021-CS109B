import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def plot_pool(img, aug_img):
    with plt.xkcd(scale=0.05):
        # The figure and axes
        fig, ax = plt.subplots(1, 3, figsize=(18, 7))
        scale = np.int((aug_img.shape[0] / img.shape[0]) * 100)
        axins = inset_axes(ax[2], width=f"{scale}%", height=f"{scale}%", loc=10)
        mpl.rcParams['text.color'] = 'k'

        plot_colour = plt.cm.bone
        # Plot with matrices
        sns.heatmap(img, annot=True, fmt="d", linewidths=.05,
                    cmap=plot_colour, cbar=False, ax=ax[1], alpha=0.5,
                    annot_kws={"fontsize": 20}, linecolor='k');

        sns.heatmap(aug_img.astype('int64'), annot=True, alpha=0.5,
                    fmt="d", linewidths=.05, cmap=plot_colour,
                    cbar=False, ax=axins, annot_kws={"fontsize": 20},
                    linecolor='k');

        # Appropriate axes
        ax[0].imshow(img, cmap='bone', alpha=0.5)
        for i in [ax[1], axins]:
            #         i.axis('off')
            i.spines['right'].set_visible(True)
            i.spines['top'].set_visible(True)
            i.spines['bottom'].set_visible(True)
            i.spines['left'].set_visible(True)
            i.get_xaxis().set_visible(False)
            i.get_yaxis().set_visible(False)
        ax[2].axis('off')
        ax[0].set_title('Input Image (3)', fontsize=20, pad=10)
        ax[1].set_title('Input array', fontsize=20, pad=10)
        axins.set_title('Output array', fontsize=20, pad=10)
#     fig.suptitle('Pooling in CNNs',fontsize=32,y=1.05)
