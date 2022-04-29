<!-- PROJECT SHIELDS -->
[![arXiv][arxiv-shield]][arxiv-url]
[![DOI][doi-shield]][doi-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

# [Port-Hamiltonian Dynamic Mode Decomposition][arxiv-url]

We present a novel physics-informed system identification method to construct a passive linear time-invariant system. In more detail, for a given quadratic energy functional, measurements of the input, state, and output of a system in the time domain, we find a realization that approximates the data well while guaranteeing that the energy functional satisfies a dissipation inequality. To this end, we use the framework of port-Hamiltonian (pH) systems and modify the dynamic mode decomposition to be feasible for continuous-time pH systems. We propose an iterative numerical method to solve the corresponding least-squares minimization problem. We construct an effective initialization of the algorithm by studying the least-squares problem in a weighted norm, for which we present the analytical minimum-norm solution. The efficiency of the proposed method is demonstrated with several numerical examples.

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#citing">Citing</a>
    </li>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#documentation">Documentation</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

## Citing
If you use this project for academic work, please consider citing our
[publication][arxiv-url]:

    R. Morandin, J. Nicodemus, and B. Unger
    Port-Hamiltonian Dynamic Mode Decomposition
    ArXiv e-print 2204.13474, 2022.

## Installation
A python environment is required with at least **Python 3.10**.

Install dependencies via `pip`:
   ```sh
   pip install -r requirements.txt
   ```

<!-- USAGE EXAMPLES -->
## Usage

The executable script `main.py` executes the `pHDMD` algorithm for the current configuration, defined in `config.py`. Both files are located in `src`.

<!-- USAGE EXAMPLES -->
## Documentation

Documentation is available [online][docs-url]
or you can build it yourself from inside the `docs` directory
by executing:

    make html

This will generate HTML documentation in `docs/build/html`.
It is required to have the `sphinx` dependencies installed. This can be done by
   
    pip install -r requirements.txt
   
within the `docs` directory.


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- CONTACT -->
## Contact
Jonas Nicodemus - jonas.nicodemus@simtech.uni-stuttgart.de

Benjamin Unger - benjamin.unger@simtech.uni-stuttgart.de\
Riccardo Morandin - morandin@math.tu-berlin.de 

Project Link: [https://github.com/Jonas-Nicodemus/phdmd][project-url]

[license-shield]: https://img.shields.io/github/license/Jonas-Nicodemus/phdmd.svg?style=for-the-badge
[license-url]: https://github.com/Jonas-Nicodemus/phdmd/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/jonas-nicodemus-a34931209/
[doi-shield]: https://img.shields.io/badge/DOI-10.5281%20%2F%20zenodo.6497497-blue.svg?style=for-the-badge
[doi-url]: https://doi.org/10.5281/zenodo.6497497
[arxiv-shield]: https://img.shields.io/badge/arXiv-2204.13474-b31b1b.svg?style=for-the-badge
[arxiv-url]: https://arxiv.org/abs/2204.13474
[project-url]:https://github.com/Jonas-Nicodemus/phdmd
[docs-url]:https://jonas-nicodemus.github.io/phdmd/