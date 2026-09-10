# DQN design of a periodic red filter

A deep Q-network changes the period, slit width, and metal thickness of a periodic slit. Meep FDTD calculates the electromagnetic response used to evaluate each design.

## Setup

```sh
git clone https://github.com/Jhnnn001/RL-metasurface.git
cd RL-metasurface
conda env create -f environment.yml
conda activate meep-dqn
```

## Run

```sh
python test_project.py
python test_project.py --fdtd
```

`dqn.py` contains the state, action, and Bellman update. `fdtd.py` contains the Meep geometry and flux calculation.

## References

- Elsawy et al., [Global optimization of metasurface designs using statistical learning methods](https://doi.org/10.1038/s41598-019-53878-9).
- [Meep Python tutorial](https://meep.readthedocs.io/en/latest/Python_Tutorials/Basics/).
