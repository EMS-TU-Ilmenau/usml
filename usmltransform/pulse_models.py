import numpy as np


def compute_pulse(params):
    # get parameters
    tPulse = params['tPulse']
    fCarrier = params['fCarrier']
    B = params['B']
    fS = params['fS']
    # calculate pulse length in samples
    N_pulse = int(np.round(tPulse * fS))
    # time vector
    if np.mod(N_pulse, 2) == 0:
        t_vec = (np.linspace(0, N_pulse, N_pulse, endpoint=False) - N_pulse / 2) / fS
    else:
        t_vec = (np.linspace(0, N_pulse-1, N_pulse, endpoint=True) - (N_pulse-1) / 2) / fS

    return np.exp(-B**2 * t_vec**2) * np.cos(2 * np.pi * fCarrier * t_vec)
