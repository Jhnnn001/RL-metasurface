"""Small checks for the actual FDTD model and the DQN update rules."""
import argparse

import numpy as np
import torch


def check_dqn():
    from dqn import bellman_target, move, valid_actions

    targets = bellman_target(
        torch.tensor([2., 2.]), torch.tensor([3., 3.]),
        torch.tensor([False, True]), gamma=0.9,
    )
    torch.testing.assert_close(targets, torch.tensor([4.7, 2.]))
    assert valid_actions((0, 0, 0)).tolist() == [False, True, False, True, False, True]
    assert move((0, 0, 0), 1) == (1, 0, 0)
    try:
        move((0, 0, 0), 0)
    except ValueError:
        pass
    else:
        raise AssertionError('Out-of-bounds action was accepted')
    print('PASS DQN terminal targets and boundary actions')


def check_fdtd():
    from fdtd import spectrum

    empty = spectrum(0.4, 0.1, 0.1, structure='air', resolution=80)
    assert np.max(np.abs(empty['R'])) < 1e-6
    assert np.max(np.abs(empty['T'] - 1)) < 1e-6
    glass = spectrum(0.4, 0.1, 0.1, structure='glass', resolution=80)
    assert np.max(np.abs(glass['R'] - 0.04)) < 0.01
    assert np.max(np.abs(glass['T'] - 0.96)) < 0.01
    print('PASS empty-cell and Fresnel-interface FDTD checks',
          float(np.max(np.abs(glass['R'] - 0.04))),
          float(np.max(np.abs(glass['T'] - 0.96))))
    try:
        spectrum(0.4, 0.5, 0.1)
    except ValueError:
        pass
    else:
        raise AssertionError('Slit wider than its period was accepted')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fdtd', action='store_true', help='also run physical Meep checks')
    args = parser.parse_args()
    check_dqn()
    if args.fdtd:
        check_fdtd()
