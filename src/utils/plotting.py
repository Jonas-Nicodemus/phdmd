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


def trajectories_plot(T, Y, Y_dmd=None, zoom=None, label='y', train=False):
    """
    Creates plot of a trajectory (input, state, output, ...).

    Parameters
    ----------
    T : numpy.ndarray
        Time steps.
    Y : numpy.ndarray
        Trajectory value at the time steps.
    Y_dmd : numpy.ndarray, optional
        Approximated trajectory value at the time steps. Default `None`.
    zoom : int, optional
        Number of samples to plot. Default `None`.
    label : str, optional
        Label of the trajectory. Default 'y'.
    train : bool, optional
        Flag if trajectory is related to the training.
    """
    fraction = 1 / 2 if config.save_results else 1

    for i, y in enumerate(Y):
        label_ = f'{label}_{i + 1}' if Y.shape[0] > 1 else f'{label}'
        title = f'Training ${label_}$' if train else f'Testing ${label_}$'
        match label:
            case 'y':
                ylabel = 'Output'
            case 'u':
                ylabel = 'Input'
            case _:
                ylabel = ''

        fig = new_fig(fraction=fraction)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(title)
        if zoom is not None:
            T = T[:zoom]
        ax.set(xlim=[np.min(T), np.max(T)])
        ax.set(xlabel='Time (s)', ylabel=ylabel)
        if zoom is not None:
            ax.plot(T, Y[i, :zoom], label=f'${label_}$')
        else:
            ax.plot(T, Y[i], label=f'${label_}$')

        if Y_dmd is not None:
            dmd_label = r'\widetilde{' + label + '}'
            dmd_ylabel = f'{dmd_label}_{i + 1}' if Y.shape[0] > 1 else f'{dmd_label}'
            if zoom is not None:
                ax.plot(T, Y_dmd[i,:zoom], ls='--', label=f'${dmd_ylabel}$')
            else:
                ax.plot(T, Y_dmd[i], ls='--', label=f'${dmd_ylabel}$')

        ax.legend(loc='best')

        if config.save_results:
            assert config.figures_path is not None and config.exp_id is not None
            ax.set_title('')
            fig.tight_layout()
            filename = f'{config.exp_id}_{label_}_train' if train else f'{config.exp_id}_{label_}'
            if zoom is not None:
                filename += '_zoom'
            filename += '.pgf'
            fig.savefig(os.path.join(config.figures_path, filename))
        else:
            plt.show()


def abs_error_plot(T, Y, Y_dmd, label='y', zoom=None):
    """
    Creates plot of an absolute error between two trajectories (state, output, ...).

    Parameters
    ----------
    T : numpy.ndarray
        Time steps.
    Y : numpy.ndarray
        Trajectory value at the time steps.
    Y_dmd : numpy.ndarray, optional
        Approximated trajectory value at the time steps. Default `None`.
    label : str, optional
        Label of the trajectory. Default 'y'.
    zoom : int, optional
        Number of samples to plot. Default `None`.
    """

    assert Y.shape == Y_dmd.shape
    assert len(T) == Y.shape[1]

    fraction = 1 / 2 if config.save_results else 1

    for i, y in enumerate(Y):
        if zoom is not None:
            T = T[:zoom]

        fig = new_fig(fraction=fraction)
        ax = fig.add_subplot(1, 1, 1)
        ax.set(xlim=[np.min(T), np.max(T)])
        ax.set(xlabel='Time (s)', ylabel=f'Absolut error')
        label_ = f'$|{label}_{i + 1} - ' + r'\widetilde{' + label + '}_' + str(i + 1) + '|$' if len(Y) > 1 \
            else f'$|{label} - ' + r'\widetilde{' + label + '}|$'
        if zoom is not None:
            E = np.abs(Y[i, :zoom] - Y_dmd[i, :zoom])
        else:
            E = np.abs(Y[i] - Y_dmd[i])

        ax.semilogy(T, E, label=label_)
        ax.legend()

        if config.save_results:
            ax.set_title('')
            fig.tight_layout()
            filename = f'{config.exp_id}_error_{label}_{i + 1}'
            if zoom is not None:
                filename += '_zoom'
            filename += '.pgf'
            fig.savefig(os.path.join(config.figures_path, filename))
        else:
            plt.show()


def bode_plot(w, lti, lti_dmd, i=None, j=None):
    """
    Creates bode plot of two lti systems.

    Parameters
    ----------
    w : numpy.ndarray
        Frequencies.
    lti : pymor.models.iosys.LTIModel
        Original lti system.
    lti_dmd : pymor.models.iosys.LTIModel
        Identified lti system.
    i : int, optional
        If specified only the bode plot of the `i`,`j` input output combination is plotted. Default `None`.
    j : int, optional
        If specified only the bode plot of the `i`,`j` input output combination is plotted. Default `None`.
    """
    fig = new_fig(fraction=1, ratio=1)
    ax = fig.subplots(2 * config.m, config.m, sharex=True, squeeze=False)
    artists = lti.transfer_function.bode_plot(w, ax=ax, label='G(s)')
    artists_dmd = lti_dmd.transfer_function.bode_plot(w, ax=ax, label='\widetilde{G}(s)', ls='--')
    if not config.save_results:
        plt.show()  # can only show figures when Matplotlib is not using pgf

    if i is not None and j is not None:
        # Bode of i,j input pair

        # Magnitude
        fig = new_fig()
        ax = fig.add_subplot(111)
        ax.set(xlim=[np.min(w), np.max(w)])
        ax.loglog(*artists[2 * i, j][0].get_data(), label='$G$')
        ax.loglog(*artists_dmd[2 * i, j][0].get_data(),
                  ls='--', label=r'$\widetilde{G}$')
        ax.legend()
        ax.set_title(r'Magnitude plot of $G_{' + str(i + 1) + str(j + 1) + '}$')
        ax.set_xlabel('Frequency (rad/s)')
        ax.set_ylabel('Magnitude')

        if config.save_results:
            ax.set_title('')
            fig.tight_layout()
            fig.savefig(os.path.join(config.figures_path, f'{config.exp_id}_mag_{i + 1}_{j + 1}.pgf'))
        else:
            plt.show()

        # Phase
        fig = new_fig()
        ax = fig.add_subplot(111)
        ax.set(xlim=[np.min(w), np.max(w)])
        ax.semilogx(*artists[2 * i + 1, j][0].get_data(), label='$G$')
        ax.semilogx(*artists_dmd[2 * i + 1, j][0].get_data(),
                    ls='--', label=r'$\widetilde{G}$')
        ax.legend()
        ax.set_title(r'Phase plot of $G_{' + str(i + 1) + str(j + 1) + '}$')
        ax.set_xlabel('Frequency (rad/s)')
        ax.set_ylabel('Phase (deg)')

        if config.save_results:
            ax.set_title('')
            fig.tight_layout()
            fig.savefig(os.path.join(config.figures_path, f'{config.exp_id}_phase_{i + 1}_{j + 1}.pgf'))
        else:
            plt.show()


def magnitude_plot(w, lti):
    """
    Creates magnitude bode plot an lti system.

    Parameters
    ----------
    w : numpy.ndarray
        Frequencies.
    lti : pymor.models.iosys.LTIModel
        (Error) lti system.
    """
    fraction = 1 / 2 if config.save_results else 1

    fig = new_fig(fraction=fraction)
    ax = fig.add_subplot(1, 1, 1)
    ax.set(xlim=[np.min(w), np.max(w)])
    lti.transfer_function.mag_plot(w, ax=ax, label='Error')
    _ = ax.legend()
    if config.save_results:
        ax.set_title('')
        fig.tight_layout()
        fig.savefig(os.path.join(config.figures_path, f'{config.exp_id}_error_mag_plot.pgf'))
    else:
        plt.show()


def poles_plot(lti, lti_dmd=None):
    """
    Creates poles bode plot of one or two lti systems.

    Parameters
    ----------
    lti : pymor.models.iosys.LTIModel
        Original lti system.
    lti_dmd : pymor.models.iosys.LTIModel
        Identified lti system.
    """
    # Eigenvalues
    poles = lti.poles()
    fig, ax = plt.subplots()
    ax.plot(poles.real, poles.imag, '.', label='FOM')
    if lti_dmd is not None:
        poles_dmd = lti_dmd.poles()
        ax.plot(poles_dmd.real, poles_dmd.imag, 'x', label='dmd')

    _ = ax.set_title('Poles')
    _ = ax.legend()
    ax.set(xlabel='Re($\lambda$)', ylabel='Im($\lambda$)')

    if config.save_results:
        ax.set_title('')
        fig.tight_layout()
        fig.savefig(os.path.join(config.figures_path, f'{config.exp_id}_poles.pgf'))
    else:
        plt.show()
