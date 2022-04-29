.. pHDMD documentation master file, created by
   sphinx-quickstart on Thu Apr 21 11:21:24 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Port-Hamiltonian Dynamic Mode Decomposition
===========================================

We present a novel physics-informed system identification method to construct
a passive linear time-invariant system. In more detail, for a given quadratic
energy functional, measurements of the input, state, and output of a system
in the time domain, we find a realization that approximates the data well
while guaranteeing that the energy functional satisfies a dissipation inequality.
To this end, we use the framework of port-Hamiltonian (pH) systems and modify
the dynamic mode decomposition to be feasible for continuous-time pH systems.
We propose an iterative numerical method to solve the corresponding least-squares
minimization problem. We construct an effective initialization of the algorithm
by studying the least-squares problem in a weighted norm, for which we present
the analytical minimum-norm solution. The efficiency of the proposed method is
demonstrated with several numerical examples.

Citing
======
If you use pHDMD for academic work, please consider citing our
`publication <https://arxiv.org/abs/2204.13474>`_:

    R. Morandin, J. Nicodemus, and B. Unger
    Port-Hamiltonian Dynamic Mode Decomposition
    ArXiv e-print 2204.13474, 2022.



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules
   bibliography
   contact

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
