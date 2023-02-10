import logging
import os

import config
from data.generate import generate
from evaluation.evaluation import evaluate


def main():
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    if not os.path.exists(config.simulations_path):
        os.makedirs(config.simulations_path)

    if not os.path.exists(config.evaluations_path):
        os.makedirs(config.evaluations_path)

    # specify experiments in config.py
    experiments = config.experiments

    for exp in experiments:
        logging.info(f'Experiment: {exp.name}')
        lti_dict = {}

        n = exp.fom.order
        H = exp.H

        logging.info(f'State dimension n = {n}')
        logging.info(f'Step size delta = {exp.delta:.2e}')
        lti_dict['Original'] = exp.fom

        # Generate/Load training data
        X_train, Y_train, U_train = generate(exp)

        # Plot training input, analogously output or state data
        # plot(exp.T, U_train, label='$u$', ylabel='Input', xlabel='Time (s)',
        #      fraction=config.fraction, name=exp.name + '_training_input')

        # Perform methods
        for method in exp.methods:
            lti = method(X_train, Y_train, U_train, exp.delta, H)
            lti_dict[method.name] = lti

        # Evaluation
        logging.info('Evaluate')
        evaluate(exp, lti_dict)


if __name__ == "__main__":
    main()
