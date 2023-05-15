Interpretable t-SNE
===================

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

`interpretable_tsne` is an implementation of our gradient-based attributiont technique described in our GLBIO oral presentation: 'Towards Computing Attributions for Dimensionality Reduction Techniques'. 

To replicate the experiments we performed in our presentation, go to: [MattScicluna/interpretable_tsne_experiment](https://github.com/MattScicluna/interpretable_tsne_experiment)

See [our pre-print](https://www.biorxiv.org/content/10.1101/2023.05.12.540592v1) for details of how the algorithm works.

---

Installation
------------

`interpretable_tsne` requires Python 3.8 or higher to run. It is only available on Linux operating systems.

### TestPyPi

This code can be installed via TestPyPi using the following command:

    pip install -i https://test.pypi.org/simple/ interpretable-tsne

### Source

You can install `interpretable_tsne` from source. Just clone this repository and run the following line in the root directory:

    pip install .

Run the unittests

    python -m unittest -v tests/*.py

---

Citation
--------

If you use `interpretable_tsne` in your work, please cite us:

```
    @article {Scicluna2023.05.12.540592,
        author = {Matthew Crispin Scicluna and Jean-Christophe Grenier and Raphael Poujol and Sebastien Lemieux and Julie Hussin},
        title = {Towards Computing Attributions for Dimensionality Reduction Techniques},
        elocation-id = {2023.05.12.540592},
        year = {2023},
        doi = {10.1101/2023.05.12.540592},
        publisher = {Cold Spring Harbor Laboratory},
        URL = {https://www.biorxiv.org/content/early/2023/05/14/2023.05.12.540592},
        eprint = {https://www.biorxiv.org/content/early/2023/05/14/2023.05.12.540592.full.pdf},
        journal = {bioRxiv}
    }
```

[def]: assets/synth_data_gradients.mp4