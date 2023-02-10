import os

import numpy as np
import matplotlib as mpl

from scipy.signal import sawtooth
from pymor.models.iosys import PHLTIModel, LTIModel

from algorithm.methods import IODMDMethod, PHDMDMethod, OIMethod
from model.msd import msd
from model.poro import poro


class Experiment:
    """
    Class for experiments.

    Parameters
    ----------
    name: str
        The name of the experiment.
    model: str
        The name of the model of the experiment.
    fom : function
        Function for getting port-Hamiltonian system matrices.
    u : function
        Training input function.
    T : numpy.ndarray
        Training time interval.
    x0 : numpy.ndarray
        Training initial condition.
    time_stepper : str, optional
        Name of the time stepping method. Default 'implicit_midpoint'.

    """

    def __init__(self, name, model, fom, u, T, x0, time_stepper='implicit_midpoint', u_test=None, T_test=None,
                 x0_test=None, r=None, noise=None, methods=None):
        if methods is None:
            methods = [PHDMDMethod()]

        self.name = name
        self.model = model

        H, J, R, G, P, S, N = fom()
        self.fom = PHLTIModel.from_matrices(J, R, G, P, S, N, H)

        self.H = H

        self.u = u
        self.T = T
        self.x0 = x0
        self.delta = T[1] - T[0]
        self.time_stepper = time_stepper

        if u_test is None:
            u_test = u

        if T_test is None:
            T_test = T

        if x0_test is None:
            x0_test = x0

        self.u_test = u_test
        self.T_test = T_test
        self.x0_test = x0_test

        self.r = r
        self.noise = noise

        self.methods = methods


siso_msd_exp = Experiment(
    name='SISO_MSD',
    model='msd',
    fom=lambda: msd(6, 1),
    u=lambda t: np.array([np.exp(-0.5 * t) * np.sin(t ** 2)]),
    T=np.linspace(0, 4, 101),
    x0=np.zeros(6),
    u_test=lambda t: np.array([sawtooth(2 * np.pi * 0.5 * t)]),
    T_test=np.linspace(0, 10, 251),
    x0_test=np.zeros(6)
)

siso_msd_exp_1 = Experiment(
    name='SISO_MSD',
    model='msd',
    fom=lambda: msd(6, 1),
    u=lambda t: np.array([np.exp(-0.5 * t) * np.sin(t ** 2)]),
    T=np.linspace(0, 4, 101),
    x0=np.zeros(6),
    u_test=lambda t: np.array([sawtooth(2 * np.pi * 0.5 * t)]),
    T_test=np.linspace(0, 10, 251),
    x0_test=np.zeros(6),
    methods=[IODMDMethod(), PHDMDMethod()],
)

siso_msd_exp_2 = Experiment(
    name='SISO_MSD_small_delta',
    model='msd',
    fom=lambda: msd(6, 1),
    u=lambda t: np.array([np.exp(-0.5 * t) * np.sin(t ** 2)]),
    T=np.linspace(0, 4, 40001),
    x0=np.zeros(6),
    u_test=lambda t: np.array([sawtooth(2 * np.pi * 0.5 * t)]),
    T_test=np.linspace(0, 10, 100001),
    x0_test=np.zeros(6),
    methods=[IODMDMethod(), PHDMDMethod()],
)

siso_msd_exp_3 = Experiment(
    name='SISO_MSD_RK45',
    model='msd',
    fom=lambda: msd(6, 1),
    u=lambda t: np.array([np.exp(-0.5 * t) * np.sin(t ** 2)]),
    T=np.linspace(0, 4, 101),
    x0=np.zeros(6),
    time_stepper='RK45',
    u_test=lambda t: np.array([sawtooth(2 * np.pi * 0.5 * t)]),
    T_test=np.linspace(0, 10, 251),
    x0_test=np.zeros(6),
    methods=[IODMDMethod(), PHDMDMethod()],
)

siso_msd_exp_4 = Experiment(
    name='SISO_MSD_noisy',
    model='msd',
    fom=lambda: msd(6, 1),
    u=lambda t: np.array([np.exp(-0.5 * t) * np.sin(t ** 2)]),
    T=np.linspace(0, 4, 101),
    x0=np.zeros(6),
    u_test=lambda t: np.array([sawtooth(2 * np.pi * 0.5 * t)]),
    T_test=np.linspace(0, 10, 251),
    x0_test=np.zeros(6),
    noise=1e-4,
    methods=[OIMethod(), PHDMDMethod()],
)

mimo_msd_exp = Experiment(
    name='MIMO_MSD',
    model='msd',
    fom=lambda: msd(100, 2),
    u=lambda t: np.array([np.exp(-0.5 / 100 * t) * np.sin(1 / 100 * t ** 2),
                          np.exp(-0.5 / 100 * t) * np.cos(1 / 100 * t ** 2)]),
    T=np.linspace(0, 4 * 100, 100 * 100 + 1),
    x0=np.zeros(100),
    u_test=lambda t: np.array([sawtooth(2 * np.pi * 0.5 * t), -sawtooth(2 * np.pi * 0.5 * t)]),
    T_test=np.linspace(0, 10, 251),
    x0_test=np.zeros(100),
    methods=[OIMethod(), PHDMDMethod()],
)

poro_exp = Experiment(
    name='PORO',
    model='poro',
    fom=lambda: poro(980),
    u=lambda t: np.array([np.exp(-0.5 / 100 * t) * np.sin(1 / 100 * t ** 2),
                          np.exp(-0.5 / 100 * t) * np.cos(1 / 100 * t ** 2)]),
    T=np.linspace(0, 4 * 100, 100 * 100 + 1),
    x0=np.zeros(980),
    u_test=lambda t: np.array([sawtooth(2 * np.pi * 0.5 * t), -sawtooth(2 * np.pi * 0.5 * t)]),
    T_test=np.linspace(0, 10, 251),
    x0_test=np.zeros(980)
)

experiments = [siso_msd_exp, siso_msd_exp_1, siso_msd_exp_2, siso_msd_exp_3, siso_msd_exp_4]
# experiments = [mimo_msd_exp]
# experiments = [poro_exp]

save_results = False  # If true all figures will be saved as pdf
width_pt = 420  # Get this from LaTeX using \the\textwidth
fraction = 0.49 if save_results else 1  # Fraction of width the figure will occupy
plot_format = 'pdf'

colors = np.array(mpl.colormaps['Set1'].colors)

plots_path = os.path.join('../plots')
data_path = os.path.join('../data')
simulations_path = os.path.join(data_path, 'simulations')
evaluations_path = os.path.join(data_path, 'evaluations')

force_simulation = False  # If true the simulation will be forced to run again
