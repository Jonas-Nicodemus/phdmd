import logging

import numpy as np

import config
from data.generate import sim
from utils.plotting import plot


def h_norm(lti_ref, lti_approx, label, compute_hinf=True):
    """
    Compute the H2 and Hinf norm of the error between the reference and the approximated system.

    Parameters
    ----------
    lti_ref : pymor.models.iosys.LTIModel
        The reference system.
    lti_approx : pymor.models.iosys.LTIModel
        The approximated system.
    label : str
        Label for the approximated system.
    compute_hinf : bool, optional
        If True, the Hinf norm is computed. Default True.

    Returns
    -------
    h2_norms : numpy.ndarray
        The H2 norm of the error.
    hinf_norms : numpy.ndarray
        The Hinf norm of the error.
    """
    if not isinstance(lti_approx, list):
        lti_approx = [lti_approx]

    if not isinstance(label, list):
        label = [label]

    logging.info('H norm errors:')
    h2_ref = lti_ref.h2_norm()

    if compute_hinf:
        hinf_ref = lti_ref.hinf_norm()

    h2_norms = np.zeros(len(lti_approx))
    hinf_norms = np.zeros(len(lti_approx))

    for i, lti_appr in enumerate(lti_approx):
        try:
            lti_error = lti_ref - lti_appr
        except AssertionError:
            lti_error = lti_ref - lti_appr.to_continuous()
        logging.info(label[i])

        h2 = lti_error.h2_norm()
        h2 = h2 / h2_ref
        logging.info(f'Relative H2 error: {h2:.2e}')
        h2_norms[i] = h2

        if compute_hinf:
            hinf = lti_error.hinf_norm()
            hinf = hinf / hinf_ref
            logging.info(f'Relative Hinf error: {hinf:.2e}')
            hinf_norms[i] = hinf

    return h2_norms, hinf_norms


def evaluate(exp, lti_dict):
    """
    Evaluate the experiment for given approximated systems.
    Generates plots and computes the H2 and (optional) Hinf norm of the error.

    Parameters
    ----------
    exp : Experiment
        The experiment to evaluate.
    lti_dict : dict
        Dictionary of approximated systems.
    """
    Y_list = []
    Y_error_list = []
    lti_error_list = []
    labels = []

    lti_list = list(lti_dict.values())
    labels_i = list(lti_dict.keys())
    labels += labels_i
    for j in range(len(lti_list)):
        logging.info(f'Evaluate {labels_i[j]}')

        # Simulate for testing input
        U, X, Y = sim(lti_list[j], exp.u_test, exp.T_test, exp.x0_test)
        Y_list.append(Y)

        if j > 0:
            Y_error = np.abs(Y_list[0] - Y_list[j])
            Y_error_list.append(Y_error)
            try:
                lti_error = lti_list[0] - lti_list[j]
            except AssertionError:
                lti_error = lti_list[0] - lti_list[j].to_continuous()

            lti_error_list.append(lti_error)

    # Trajectories
    # plot(exp.T_test, U, label='$u$', ylabel='Input', xlabel='Time (s)', legend='upper right',
    #      fraction=config.fraction, name=exp.name + '_testing_input')

    ls = len(labels) * ['--']
    ls[0] = '-'
    testing_output_labels = ['$y$', *[r'$\widetilde{y}_{\mathrm{' + l + '}}$' for l in labels[1:]]]
    plot(exp.T_test, Y_list, label=testing_output_labels, c=config.colors, xlabel='Time',
         # ylim=[-1.5, 0.5],
         ylabel='Testing output', name=f'{exp.name}_testing_output', ls=ls, fraction=config.fraction)

    # Absolute Error of the trajectories
    plot(exp.T_test, Y_error_list, label=labels[1:], c=config.colors[1:], yscale='log', xlabel='Time',
         ylabel='Absolute error',
         name=f'{exp.name}_abs_error', fraction=config.fraction)

    # H-norm calculation fails for the poro benchmark system
    if exp.model != 'poro':
        h_norm(lti_list[0], lti_list[1:], labels[1:])
