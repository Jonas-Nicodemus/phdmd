import logging
import os
import config

import numpy as np

from algorithm.phdmd import phdmd
from system.phlti import PHLTI
from utils.norm import lnorm
from utils.plotting import trajectories_plot, abs_error_plot, bode_plot, magnitude_plot, poles_plot


def main():
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    if config.save_results:
        import matplotlib
        matplotlib.use("pgf")
        matplotlib.rcParams.update({
            "pgf.texsystem": "pdflatex",
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
        })

        if not os.path.exists(config.figures_path):
            os.makedirs(config.figures_path)

    # Initialize the original ph system
    E, J, R, G, P, D, Q = config.ph_matrices()

    ph = PHLTI(E, J, R, G, P, D, Q)
    lti = ph.to_lti(matrices=False)

    # Generate training data
    U_train, X_train, Y_train = ph.sim(config.u, config.T, config.x0)

    trajectories_plot(config.T, U_train, label='u', train=True)
    trajectories_plot(config.T, Y_train, train=True)

    # Perform pHDMD
    J_dmd, R_dmd = phdmd(X_train, Y_train, U_train, config.delta,
                         E=ph.E, max_iter=100)
    ph_dmd = PHLTI(ph.E, J_dmd, R_dmd)

    # Testing
    U_train_dmd, X_train_dmd, Y_train_dmd = ph_dmd.sim(config.u, config.T, config.x0)
    U_test, X_test, Y_test = ph.sim(config.u_test, config.T_test, config.x0)
    U_dmd, X_dmd, Y_dmd = ph_dmd.sim(config.u_test, config.T_test, config.x0)

    # Error lti system
    lti_dmd = ph_dmd.to_lti(matrices=False)
    lti_error = lti - lti_dmd

    # Trajectories
    trajectories_plot(config.T_test, U_test, label='u')
    trajectories_plot(config.T_test, Y_test, Y_dmd)

    # Absolute Error of the trajectories
    abs_error_plot(config.T_test, Y_test, Y_dmd)

    # Only first part of the trajectories
    trajectories_plot(config.T_test, Y_test, Y_dmd, zoom=100)
    abs_error_plot(config.T_test, Y_test, Y_dmd, zoom=100)

    # Bode
    w = np.logspace(-1, 3, 100)
    bode_plot(w, lti, lti_dmd)

    # Magnitude error system
    magnitude_plot(w, lti_error)

    # Poles
    poles_plot(lti, lti_dmd)

    # Norms
    l2_train = lnorm(Y_train - Y_train_dmd, p=2, tau=config.delta)
    l2_rel_train = l2_train / lnorm(Y_train, p=2, tau=config.delta)
    logging.info(f'Relative L2 error (Training data): {l2_rel_train:.2e}')

    linf_train = lnorm(Y_train - Y_train_dmd, p=np.inf, tau=config.delta)
    linf_rel_train = linf_train / lnorm(Y_train, p=np.inf, tau=config.delta)
    logging.info(f'Relative Linf error (Training data): {linf_rel_train:.2e}')

    l2 = lnorm(Y_test - Y_dmd, p=2, tau=config.delta)
    l2_rel = l2 / lnorm(Y_test, p=2, tau=config.delta)
    logging.info(f'Relative L2 error: {l2_rel:.2e}')

    linf_train = lnorm(Y_test - Y_dmd, p=np.inf, tau=config.delta)
    linf_rel = linf_train / lnorm(Y_test, p=np.inf, tau=config.delta)
    logging.info(f'Relative Linf error: {linf_rel:.2e}')

    # H-norm calculation fails for the poro benchmark system
    if config.model != 'poro':
        h2 = lti_error.h2_norm()
        h2_rel = h2 / lti.h2_norm()
        logging.info(f'Relative H2 error: {h2_rel:.2e}')

        hinf = lti_error.hinf_norm()
        hinf_rel = hinf / lti.hinf_norm()
        logging.info(f'Relative Hinf error: {hinf_rel:.2e}')


if __name__ == "__main__":
    main()
