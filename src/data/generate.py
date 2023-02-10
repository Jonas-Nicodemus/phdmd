import logging
import os

import numpy as np

import config
from discretization.discretization import discretize

from pymor.algorithms.to_matrix import to_matrix


def sim(lti, U, T=None, x0=None, method='implicit_midpoint', return_dXdt=False):
    """
    Simulate a linear time invariant system.

    Parameters
    ----------
    lti : pymor.models.iosys.PHLTIModel
        The LTI system to simulate.
    U : np.ndarray or callable
        The input signal. If callable, it must take a single argument T and return a 2D array of shape (dim_input, len(T)).
    T : np.ndarray
        The time instants at which to compute the data.
    x0 : np.ndarray
        The initial state.
    method : str
        The method to use for the simulation. See :func:`discretize` for a list of available methods.
    return_dXdt : bool
        Whether to return the time derivative of the state.

    Returns
    -------
    U : np.ndarray
        The input signal.
    X : np.ndarray
        The state data.
    Y : np.ndarray
        The output data.
    """
    if T is None:
        assert isinstance(U, np.ndarray)
        T = np.linspace(0, len(U))

    if x0 is None:
        x0 = np.zeros(lti.order)

    if lti.sampling_time > 0:
        if not isinstance(U, np.ndarray):
            U = U(T)
            if U.ndim < 2:
                U = U[np.newaxis, :]
        return sim_disc(lti, U, x0)

    return discretize(lti, U, T, x0, method, return_dXdt)


def sim_disc(lti_disc, U, x0):
    """
    Simulate a discrete-time linear time invariant system.

    Parameters
    ----------
    lti_disc : pymor.models.iosys.LTIModel
        The discrete-time LTI system to simulate.
    U : np.ndarray
        The input signal.
    x0 : np.ndarray
        The initial state.

    Returns
    -------
    U : np.ndarray
        The input signal.
    X : np.ndarray
        The state data.
    Y : np.ndarray
        The output data.
    """
    X = np.zeros((lti_disc.order, U.shape[1] + 1))
    Y = np.zeros((lti_disc.dim_output, U.shape[1]))

    X[:, 0] = x0

    A = to_matrix(lti_disc.A)
    B = to_matrix(lti_disc.B)
    C = to_matrix(lti_disc.C)
    D = to_matrix(lti_disc.D)

    for i in range(U.shape[1]):
        X[:, i + 1] = A @ X[:, i] + B @ U[:, i]
        Y[:, i] = C @ X[:, i] + D @ U[:, i]

    return U, X, Y


def generate(exp):
    """
    Generate or load training data for a given experiment.

    Parameters
    ----------
    exp : Experiment
        The experiment for which to generate the training data.

    Returns
    -------
    X_train : np.ndarray
        The state data.
    Y_train : np.ndarray
        The output data.
    U_train : np.ndarray
        The input data.
    """
    if not os.path.exists(config.simulations_path + '/' + exp.name + '_sim.npz') or config.force_simulation:
        logging.info('Simulate FOM')
        U_train, X_train, Y_train = sim(exp.fom, exp.u, exp.T, exp.x0,
                                        method=exp.time_stepper)
        np.savez(os.path.join(config.simulations_path, exp.name + '_sim'),
                 U_train=U_train, X_train=X_train, Y_train=Y_train)
    else:
        logging.info('Load training data')
        npzfile = np.load(os.path.join(config.simulations_path, exp.name + '_sim.npz'))
        U_train = npzfile['U_train']
        X_train = npzfile['X_train']
        Y_train = npzfile['Y_train']

    if exp.noise is not None:
        logging.info(f'Noise: {exp.noise:.2e}')
        np.random.seed(42)
        X_train += exp.noise * np.random.standard_normal(size=X_train.shape)
        Y_train += exp.noise * np.random.standard_normal(size=Y_train.shape)

    return X_train, Y_train, U_train
