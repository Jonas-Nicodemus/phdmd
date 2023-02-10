import os

import matplotlib.pyplot as plt
import numpy as np

import config


def fig_size(width_pt, fraction=1, ratio=(5 ** .5 - 1) / 2, subplots=(1, 1)):
    """
    Returns the width and heights in inches for a matplotlib figure.

    Parameters
    ----------
    width_pt : float
        Document width in points, in latex can be determined with `\showthe\textwidth`.
    fraction : float, optional
        The fraction of the width with which the figure will occupy. Default 1.
    ratio : float, optional
        Ratio of the figure. Default is the golden ratio.
    subplots : tuple, optional
        The shape of subplots.

    Returns
    -------
    fig_width_in : float
        Width of the figure in inches.
    fig_height_in : float
        Height of the figure in inches.
    """
    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = ratio

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return fig_width_in, fig_height_in


def new_fig(width_pt=420, fraction=1 / 2, ratio=(5 ** .5 - 1) / 2, subplots=(1, 1)):
    """
    Creates new instance of a `matplotlib.pyplot.figure` by using the `fig_size` function.

    Parameters
    ----------
    width_pt : float, optional
        Document width in points, in latex can be determined with `\showthe\textwidth`.
        Default is 420.
    fraction : float, optional
        The fraction of the width with which the figure will occupy. Default 1.
    ratio : float, optional
        Ratio of the figure. Default is the golden ratio.
    subplots : tuple, optional
        The shape of subplots.

    Returns
    -------
    fig : matplotlib.pyplot.figure
        The figure.
    """

    fig = plt.figure(figsize=fig_size(width_pt, fraction, ratio, subplots))
    return fig


def plot(X, Y, label=None, ls=None, marker=None, c=None, markevery=None,
         xlabel=None, ylabel=None, legend='best', grid=False, xscale='linear', yscale='linear',
         xlim=None, ylim=None, subplots=True,
         fraction=1, ratio=(5 ** .5 - 1) / 2, name=None):
    """
    Plots a trajectory.

    Parameters
    ----------
    X : np.ndarray
        The x-axis values.
    Y : np.ndarray
        The y-axis values.
    label : np.ndarray, optional
        The labels of the lines.
    ls : np.ndarray, optional
        The line styles.
    marker : np.ndarray, optional
        The markers.
    c : np.ndarray, optional
        The colors.
    markevery : int, optional
        The number of points to skip between markers.
    xlabel : str, optional
        The x-axis label.
    ylabel : str, optional
        The y-axis label.
    legend : str, optional
        The legend location.
    grid : bool, optional
        Whether to show the grid.
    xscale : str, optional
        The x-axis scale.
    yscale : str, optional
        The y-axis scale.
    xlim : tuple, optional
        The x-axis limits.
    ylim : tuple, optional
        The y-axis limits.
    subplots : bool, optional
        Whether to plot as subplots or in one plot.
    fraction : float, optional
        The fraction of the width with which the figure will occupy. Default 1.
    ratio : float, optional
        Ratio of the figure. Default is the golden ratio.
    name : str, optional
        The name of the figure.
    """
    if isinstance(Y, list):
        Y = np.array(Y)

    if Y.ndim == 1:
        Y = Y[np.newaxis, np.newaxis, :]

    if Y.ndim == 2:
        Y = Y[np.newaxis, :, :]

    if isinstance(X, list):
        X = np.array(X)

    if X.ndim == 1:
        X = np.tile(X, (Y.shape[0], 1))

    if label is None:
        legend = False
        label = np.empty(Y.shape[:2], dtype='object')

    if isinstance(label, str):
        label = np.array([label])

    if isinstance(label, list):
        label = np.array(label)

    if label.ndim == 1:
        label = label[:, np.newaxis]
        label = np.tile(label, (1, Y.shape[1]))
        # for i in range(label.shape[0]):
        #     if label.shape[1] > 1:
        #         for j in range(label.shape[1]):
        #             label[i, j] = f'{label[i, j]}_{j+1}'
        #     else:
        #         label[i, 0] = f'{label[i, 0]}'

    if ls is None:
        ls = np.empty(Y.shape[:2], dtype='object')

    if isinstance(ls, list):
        ls = np.array(ls)

    if ls.ndim == 1:
        ls = ls[:, np.newaxis]
        ls = np.tile(ls, (1, Y.shape[1]))

    if marker is None:
        marker = np.empty(Y.shape[:2], dtype='object')

    if isinstance(marker, list):
        marker = np.array(marker)

    if marker.ndim == 1:
        marker = marker[:, np.newaxis]
        marker = np.tile(marker, (1, Y.shape[1]))

    if c is None:
        c = np.empty(Y.shape[:2], dtype='object')

    if isinstance(c, list):
        c = np.array(c)

    if c.ndim == 1:
        c = c[np.newaxis, :]
        c = np.tile(c, (Y.shape[0], 1))

    if c.dtype == np.float64 and c.ndim == 2:
        c = c[:, np.newaxis, :]
        c = np.tile(c, (1, Y.shape[1], 1))

    fig = new_fig(width_pt=config.width_pt, fraction=fraction, ratio=ratio)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(ylabel)
    if xlim is None:
        xlim = [np.min(X), np.max(X)]
    ax.set(xlim=xlim)

    if ylim is not None:
        ax.set(ylim=ylim)

    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_yscale(yscale)

    if subplots:
        figs = np.empty(Y.shape[1], dtype='object')
        axes = np.empty(Y.shape[1], dtype='object')
        figs[0] = fig
        axes[0] = ax

        for i in range(Y.shape[1]-1):
            figs[i+1] = new_fig(width_pt=config.width_pt, fraction=fraction, ratio=ratio)
            axes[i+1] = figs[i+1].add_subplot(1, 1, 1)
            axes[i+1].set_title(ylabel)
            axes[i+1].set(xlim=[np.min(X), np.max(X)])
            axes[i+1].set(xlabel=xlabel, ylabel=ylabel)
            axes[i+1].set_yscale(yscale)

    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if subplots:
                axes[j].plot(X[i], Y[i, j], label=label[i, j], ls=ls[i, j],
                             marker=marker[i,j], c=c[i, j], markevery=markevery)
            else:
                ax.plot(X[i], Y[i, j], label=label[i, j], ls=ls[i, j],
                        marker=marker[i,j], c=c[i, j], markevery=markevery)

    if legend is not None:
        ax.legend(loc=legend)
        if subplots:
            for i in range(Y.shape[1] - 1):
                axes[i + 1].legend(loc=legend)

    if grid is True:
        ax.grid(True)
        if subplots:
            for i in range(Y.shape[1] - 1):
                axes[i + 1].grid(True)

    process_fig(ax, fig, name)


def process_fig(ax, fig, name):
    """
    Shows or save the figure.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes of the figure.
    fig : matplotlib.figure.Figure
        Figure to be processed.
    name : str
        Name of the figure.
    """
    if config.save_results:
        ax.set_title('')
        fig.tight_layout(pad=0.5)
        fig.savefig(os.path.join(config.plots_path, f'{name}.{config.plot_format}'))
    else:
        plt.show()
