This code performs inference, learning, and outputs a taxonomy, as described by the following paper:
> Jonathan Bragg, Mausam, and Daniel S. Weld. 2013. [Crowdsourcing multi-label classification for taxonomy creation](https://homes.cs.washington.edu/~jbragg/files/bragg-hcomp13.pdf). In Proceedings of the First AAAI Conference on Human Computation and Crowdsourcing (HCOMP '13). Palm Springs, CA, USA.

## Install
Install [Miniconda](https://conda.io/miniconda.html) and execute the following command to install dependencies:
```
conda env create -f environment.yml -n deluge
```

## Usage
- The main script `run.py` accepts a data file as described in `preprocess.py` and uses inference methods to generate taxonomies (which are printed to stdout).
- `control.py` and `util.py` are provided as references for the infogain criterion, but adaptive item selection is not fully integrated in this release.
