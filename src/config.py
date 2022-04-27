import os

import numpy as np

from scipy.signal import sawtooth

from model.msd import msd
from model.poro import poro

save_results = False  # If true all figures will be saved as pgf plots
figures_path = os.path.join('../figures')

match 1:  # 1 - SISO MSD with few data, 2 - MIMO MSD with few data, 3 - MIMO MSD with many data, 4 - PORO
    case 1:  # SISO MSD with few data
        model = 'msd'  # msd, poro
        n = 6
        m = 1
        ph_matrices = lambda: msd(n, m)
        u = lambda t: np.exp(-0.5 * t) * np.sin(t ** 2)
        exp_id = model + f'_n_{n}_m_{m}'

        x0 = np.zeros(n)
        N = 100
        T_end = 4
        T = np.linspace(0, T_end, N)
        delta = T[1] - T[0]

        T_test = np.linspace(0, 10, 1000)
        freq = 0.5  # Hz
        u_test = lambda t: sawtooth(2 * np.pi * freq * t)

    case 2:  # MIMO MSD with few data
        model = 'msd'  # msd, poro
        n = 100
        m = 2
        ph_matrices = lambda: msd(n, m)
        factor = 100
        u = lambda t: np.array([np.exp(-0.5 * t) * np.sin(t ** 2),
                                np.exp(-0.5 * t) * np.cos(t ** 2)])
        exp_id = model + f'_n_{n}_m_{m}_short'

        x0 = np.zeros(n)
        N = 100
        T_end = 4
        T = np.linspace(0, T_end, N)
        delta = T[1] - T[0]

        T_test = np.linspace(0, 10, 1000)
        freq = 0.5  # Hz
        u_test = lambda t: np.array([sawtooth(2 * np.pi * freq * t), - sawtooth(2 * np.pi * freq * t)])

    case 3:  # MIMO MSD with many data
        model = 'msd'  # msd, poro
        n = 100
        m = 2
        ph_matrices = lambda: msd(n, m)
        factor = 100
        u = lambda t: np.array([np.exp(-0.5 / factor * t) * np.sin(1 / factor * t ** 2),
                                np.exp(-0.5 / factor * t) * np.cos(1 / factor * t ** 2)])
        exp_id = model + f'_n_{n}_m_{m}_long'

        x0 = np.zeros(n)
        N = 100 * factor
        T_end = 4 * factor
        T = np.linspace(0, T_end, N)
        delta = T[1] - T[0]

        T_test = np.linspace(0, 10, 1000)
        freq = 0.5  # Hz
        u_test = lambda t: np.array([sawtooth(2 * np.pi * freq * t), - sawtooth(2 * np.pi * freq * t)])

    case 4:  # PORO
        model = 'poro'  # msd, poro
        n = 980
        m = 2
        ph_matrices = lambda: poro(n)
        factor = 100
        u = lambda t: np.array([np.exp(-0.5 / factor * t) * np.sin(1 / factor * t ** 2),
                                np.exp(-0.5 / factor * t) * np.cos(1 / factor * t ** 2)])
        exp_id = model + f'_n_{n}_m_{m}'

        x0 = np.zeros(n)
        N = 100 * factor
        T_end = 4 * factor
        T = np.linspace(0, T_end, N)
        delta = T[1] - T[0]

        T_test = np.linspace(0, 10, 10000)
        freq = 0.5  # Hz
        u_test = lambda t: np.array([sawtooth(2 * np.pi * freq * t), - sawtooth(2 * np.pi * freq * t)])
