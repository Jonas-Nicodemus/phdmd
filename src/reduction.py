import logging
import os

import numpy as np

import pymor.core.logger

from tqdm import tqdm

from pymor.algorithms.to_matrix import to_matrix
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.reductors.basic import LTIPGReductor
from pymor.basic import project

import config

from data.generate import generate
from evaluation.evaluation import h_norm
from utils.plotting import plot


def main():
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    pymor.core.logger.set_log_levels({'pymor': 'WARNING'})

    if not os.path.exists(config.simulations_path):
        os.makedirs(config.simulations_path)

    if not os.path.exists(config.evaluations_path):
        os.makedirs(config.evaluations_path)

    exp = config.mimo_msd_exp
    logging.info(f'Experiment: {exp.name}')

    reduced_orders = np.array(range(2, 102, 2))
    noise_levels = np.array([None, 1e-4, 1e-6])

    experiment_name = f'{exp.name}_h_norms'

    num_methods = len(exp.methods) + 1

    h2_norms = np.zeros((len(reduced_orders), num_methods * len(noise_levels)))
    hinf_norms = np.zeros((len(reduced_orders), num_methods * len(noise_levels)))
    labels = np.empty(num_methods * len(noise_levels), dtype=object)

    for i, noise in enumerate(noise_levels):

        if noise is not None:
            experiment_name_i = experiment_name + f'_noise_{noise:.0e}'
        else:
            experiment_name_i = experiment_name

        if not os.path.exists(config.evaluations_path + '/' + f'{experiment_name_i}.npz'):

            n = exp.fom.order
            H = exp.H
            logging.info(f'State dimension n={n}')

            # Set noise for the experiment
            exp.noise = noise
            # Generate/Load training data
            X_train, Y_train, U_train = generate(exp)

            h2_norms_i = np.zeros((len(reduced_orders), num_methods))
            hinf_norms_i = np.zeros((len(reduced_orders), num_methods))

            # POD
            VV = np.linalg.svd(X_train, full_matrices=False)[0]

            for j, r in enumerate(tqdm(reduced_orders)):
                lti_dict = {}

                V = NumpyVectorSpace.from_numpy(VV[:, :r].T, id='STATE')

                # POD with Q lhs
                pg_reductor = LTIPGReductor(exp.fom.to_lti(), V, V)
                lti_pod = pg_reductor.reduce()
                lti_dict['POD'] = lti_pod

                # Transofrm data
                X_train_red = to_matrix(project(NumpyMatrixOperator(X_train, range_id='STATE'), V, None))
                H_red = to_matrix(project(NumpyMatrixOperator(H, source_id='STATE', range_id='STATE'), V, V))

                # Perform methods
                for method in exp.methods:
                    lti = method(X_train_red, Y_train, U_train, exp.delta, H_red)
                    lti_dict[method.name] = lti

                h2, hinf = h_norm(exp.fom, list(lti_dict.values()), list(lti_dict.keys()), compute_hinf=True)
                h2_norms_i[j] = h2
                hinf_norms_i[j] = hinf

            labels_i = np.array(list(lti_dict.keys()))
            np.savez(os.path.join(config.evaluations_path, f'{experiment_name_i}'), h2_norms=h2_norms_i,
                     hinf_norms=hinf_norms_i, labels=labels_i, reduced_orders=reduced_orders)
        else:
            npzfile = np.load(os.path.join(config.evaluations_path, f'{experiment_name_i}.npz'))
            h2_norms_i = npzfile['h2_norms']
            hinf_norms_i = npzfile['hinf_norms']
            labels_i = npzfile['labels']
            reduced_orders = npzfile['reduced_orders']

        if noise is not None:
            labels_i = np.core.defchararray.add(labels_i, f' ($s={noise_levels[i]:.0e}$)')

        labels[i * num_methods:(i + 1) * num_methods] = labels_i
        h2_norms[:, i * num_methods:(i + 1) * num_methods] = h2_norms_i
        hinf_norms[:, i * num_methods:(i + 1) * num_methods] = hinf_norms_i

    c = config.colors[:3]
    c = np.tile(c, (len(noise_levels), 1))

    ls = np.array(['-', '-', '-', '--', '--', '--', ':', ':', ':'])

    markers = np.array(['o', 's', 'D'])
    markers = np.tile(markers, len(noise_levels))

    markevery = 10

    plot(reduced_orders, h2_norms.T[:, np.newaxis, :], label=labels,
         c=c, ls=ls, marker=markers, markevery=markevery,
         yscale='log', ylabel='$\mathcal{H}_2$ error', grid=True, subplots=False,
         xlabel='Reduced order', fraction=1, name=f'{experiment_name}_h2')


if __name__ == "__main__":
    main()
