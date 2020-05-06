import numpy as np
from fractions import Fraction


class FilterChan:
    def __init__(self, fc, n_cbb):
        self._t_non_overlap = 0  # if using non-overlapping clock. Not tested
        self._N = 4  # number of phases
        self._C_unit = 4.2e-12  # from chip implementation
        self._R_BB = 1 / (15e3 * self._C_unit)  # from chip TCAS sim
        self._N_CBB = n_cbb
        self._fsc = 3 * fc
        self._tau1 = self._R_BB * self._C_unit * (8 * self._N_CBB) / (8 + self._N_CBB)
        self._tau2 = self._R_BB * self._C_unit * (8 * (self._N_CBB + 1)) / (8 + (self._N_CBB + 1))
        self._pulse_width = (1. / (12 * fc)) - self._t_non_overlap  # in time(seconds)

    def generate_subband(self, signal, fs):
        """
        Generates the filtered signal for a single channel
        :param signal:
        :param fs: sample rate. Assumption here is that the signal has been sufficiently upsampled before calling this function
        :return:
        """
        # signal = cast(signal, 'double')  # if signal was read as an integer
        samples = np.arange(signal.shape[0])
        # create differential input
        signal_p = signal / 2
        signal_n = signal / (-2)
        dt = 1 / fs

        vcsb = np.zeros((self._N, signal.shape[0]))  # n-path cap voltages
        vo = np.zeros((2, signal.shape[0]))  # output voltages(plus minus,:)
        time = dt * samples

        posedge_cnt = np.ceil(time / (self._pulse_width + self._t_non_overlap))  # clk rising edge count
        phi_sb_sel = (np.ceil(posedge_cnt / 3) % 4) + 1
        eps = 1e-6 # used for my rounding scheme. Probably better ways to do this
        phi_sc = 1 - np.round(((posedge_cnt + 1) % 4) / 4 + eps)
        # phi_sc = time % (1 / self._fsc) >= 1 / (2 * self._fsc)  # switch-cap clock
        # phi_sc = np.ceil(time / (1 / (2 * self._fsc)))

        state = np.ones(signal.shape)  # states over time

        state[np.logical_and(phi_sb_sel == 4, phi_sc == 1)] = 1
        state[np.logical_and(phi_sb_sel == 4, phi_sc == 0)] = 2
        state[np.logical_and(phi_sb_sel == 1, phi_sc == 0)] = 3
        state[np.logical_and(phi_sb_sel == 1, phi_sc == 1)] = 4
        state[np.logical_and(phi_sb_sel == 2, phi_sc == 0)] = 5
        state[np.logical_and(phi_sb_sel == 2, phi_sc == 1)] = 6
        state[np.logical_and(phi_sb_sel == 3, phi_sc == 1)] = 7
        state[np.logical_and(phi_sb_sel == 3, phi_sc == 0)] = 8

        vop_target = vom_target = 0

        for i, t in enumerate(time, start=1):
            if i >= state.shape[0]: # because we started at i=1
                break
            if state[i] < state[i - 1]: # can go back a state unless you wrap around
                if state[i - 1] == 8:
                    state[i] = 1
                else:
                    state[i] = state[i - 1]
            vcsb[:, i] = vcsb[:, i - 1]
            if state[i] == 1:  # self._tau1 state
                if state[i] != state[i - 1]:
                    vop_target = (self._N_CBB * vcsb[3, i] + 8 * vo[0, i - 1]) / (self._N_CBB + 8)
                    vom_target = (self._N_CBB * vcsb[1, i] + 8 * vo[1, i - 1]) / (self._N_CBB + 8)
                vo[0, i] = vop_target - (vop_target - vo[0, i - 1]) * np.exp(-dt / self._tau1)
                vcsb[3, i] = vop_target - (vop_target - vcsb[3, i]) * np.exp(-dt / self._tau1)
                vo[1, i] = vom_target - (vom_target - vo[1, i - 1]) * np.exp(-dt / self._tau1)
                vcsb[1, i] = vom_target - (vom_target - vcsb[1, i]) * np.exp(-dt / self._tau1)

            elif state[i] == 2:  # self._tau2 state
                if state[i] != state[i - 1]:
                    vcsb[3, i] = (signal_p[i - 1] + self._N_CBB * vcsb[3, i - 1]) / (self._N_CBB + 1)
                    vcsb[1, i] = (signal_n[i - 1] + self._N_CBB * vcsb[1, i - 1]) / (self._N_CBB + 1)
                    vop_target = ((self._N_CBB + 1) * vcsb[3, i] + 8 * vo[0, i - 1]) / (self._N_CBB + 1 + 8)
                    vom_target = ((self._N_CBB + 1) * vcsb[1, i] + 8 * vo[1, i - 1]) / (self._N_CBB + 1 + 8)
                vo[0, i] = vop_target - (vop_target - vo[0, i - 1]) * np.exp(-dt / self._tau2)
                vcsb[3, i] = vop_target - (vop_target - vcsb[3, i]) * np.exp(-dt / self._tau2)
                vo[1, i] = vom_target - (vom_target - vo[1, i - 1]) * np.exp(-dt / self._tau2)
                vcsb[1, i] = vom_target - (vom_target - vcsb[1, i]) * np.exp(-dt / self._tau2)

            elif state[i] == 3:  # self._tau2 state
                if state[i] != state[i - 1]:
                    vcsb[0, i] = (self._N_CBB * (vcsb[0, i - 1]) + vcsb[3, i - 1]) / (self._N_CBB + 1)
                    vcsb[2, i] = (self._N_CBB * (vcsb[2, i - 1]) + vcsb[1, i - 1]) / (self._N_CBB + 1)
                    vop_target = ((self._N_CBB + 1) * vcsb[0, i] + 8 * vo[0, i - 1]) / (self._N_CBB + 1 + 8)
                    vom_target = ((self._N_CBB + 1) * vcsb[2, i] + 8 * vo[1, i - 1]) / (self._N_CBB + 1 + 8)
                vo[0, i] = vop_target - (vop_target - vo[0, i - 1]) * np.exp(-dt / self._tau2)
                vcsb[0, i] = vop_target - (vop_target - vcsb[0, i]) * np.exp(-dt / self._tau2)
                vo[1, i] = vom_target - (vom_target - vo[1, i - 1]) * np.exp(-dt / self._tau2)
                vcsb[2, i] = vom_target - (vom_target - vcsb[2, i]) * np.exp(-dt / self._tau2)

            elif state[i] == 4:  # self._tau1 state
                if state[i] != state[i - 1]:
                    vop_target = (self._N_CBB * vcsb[0, i] + 8 * vo[0, i - 1]) / (self._N_CBB + 8)
                    vom_target = (self._N_CBB * vcsb[2, i] + 8 * vo[1, i - 1]) / (self._N_CBB + 8)
                vo[0, i] = vop_target - (vop_target - vo[0, i - 1]) * np.exp(-dt / self._tau1)
                vcsb[0, i] = vop_target - (vop_target - vcsb[0, i]) * np.exp(-dt / self._tau1)
                vo[1, i] = vom_target - (vom_target - vo[1, i - 1]) * np.exp(-dt / self._tau1)
                vcsb[2, i] = vom_target - (vom_target - vcsb[2, i]) * np.exp(-dt / self._tau1)

            elif state[i] == 5:  # self._tau2 state
                if state[i] != state[i - 1]:
                    vcsb[1, i] = (signal_p[i - 1] + self._N_CBB * vcsb[1, i - 1]) / (self._N_CBB + 1)
                    vcsb[3, i] = (signal_n[i - 1] + self._N_CBB * vcsb[3, i - 1]) / (self._N_CBB + 1)
                    vop_target = ((self._N_CBB + 1) * vcsb[3, i] + 8 * vo[0, i - 1]) / (self._N_CBB + 1 + 8)
                    vom_target = ((self._N_CBB + 1) * vcsb[1, i] + 8 * vo[1, i - 1]) / (self._N_CBB + 1 + 8)
                vo[0, i] = vop_target - (vop_target - vo[0, i - 1]) * np.exp(-dt / self._tau2)
                vcsb[3, i] = vop_target - (vop_target - vcsb[3, i]) * np.exp(-dt / self._tau2)
                vo[1, i] = vom_target - (vom_target - vo[1, i - 1]) * np.exp(-dt / self._tau2)
                vcsb[1, i] = vom_target - (vom_target - vcsb[1, i]) * np.exp(-dt / self._tau2)

            elif state[i] == 6:  # self._tau1 state
                if state[i] != state[i - 1]:
                    vop_target = (self._N_CBB * vcsb[3, i] + 8 * vo[0, i - 1]) / (self._N_CBB + 8)
                    vom_target = (self._N_CBB * vcsb[1, i] + 8 * vo[1, i - 1]) / (self._N_CBB + 8)
                vo[0, i] = vop_target - (vop_target - vo[0, i - 1]) * np.exp(-dt / self._tau1)
                vcsb[3, i] = vop_target - (vop_target - vcsb[3, i]) * np.exp(-dt / self._tau1)
                vo[1, i] = vom_target - (vom_target - vo[1, i - 1]) * np.exp(-dt / self._tau1)
                vcsb[1, i] = vom_target - (vom_target - vcsb[1, i]) * np.exp(-dt / self._tau1)

            elif state[i] == 7:  # self._tau1 state
                if state[i] != state[i - 1]:
                    vop_target = (self._N_CBB * vcsb[0, i] + 8 * vo[0, i - 1]) / (self._N_CBB + 8)
                    vom_target = (self._N_CBB * vcsb[2, i] + 8 * vo[1, i - 1]) / (self._N_CBB + 8)
                vo[0, i] = vop_target - (vop_target - vo[0, i - 1]) * np.exp(-dt / self._tau1)
                vcsb[0, i] = vop_target - (vop_target - vcsb[0, i]) * np.exp(-dt / self._tau1)
                vo[1, i] = vom_target - (vom_target - vo[1, i - 1]) * np.exp(-dt / self._tau1)
                vcsb[2, i] = vom_target - (vom_target - vcsb[2, i]) * np.exp(-dt / self._tau1)

            elif state[i] == 8:  # self._tau2 state
                if state[i] != state[i - 1]:
                    vcsb[2, i] = (self._N_CBB * vcsb[2, i - 1] + signal_p[i - 1]) / (1 + self._N_CBB)
                    vcsb[0, i] = (self._N_CBB * vcsb[0, i - 1] + signal_n[i - 1]) / (1 + self._N_CBB)
                    vop_target = ((self._N_CBB + 1) * vcsb[0, i] + 8 * vo[0, i - 1]) / (self._N_CBB + 1 + 8)
                    vom_target = ((self._N_CBB + 1) * vcsb[2, i] + 8 * vo[1, i - 1]) / (self._N_CBB + 1 + 8)
                vo[0, i] = vop_target - (vop_target - vo[0, i - 1]) * np.exp(-dt / self._tau2)
                vcsb[0, i] = vop_target - (vop_target - vcsb[0, i]) * np.exp(-dt / self._tau2)
                vo[1, i] = vom_target - (vom_target - vo[1, i - 1]) * np.exp(-dt / self._tau2)
                vcsb[2, i] = vom_target - (vom_target - vcsb[2, i]) * np.exp(-dt / self._tau2)

        subband = vo[0, :] - vo[1, :]

        # debug
        # import matplotlib.pyplot as plt
        # LLIM = 0
        # RLIM = 0.01
        # plt.subplot(4, 1, 1)
        # plt.plot(time, signal_p, time, signal_n)
        # plt.gca().set_xlim(left=LLIM, right=RLIM)
        # plt.subplot(4, 1, 2)
        # plt.plot(time, np.transpose(vo))
        # plt.gca().set_xlim(left=LLIM, right=RLIM)
        # plt.subplot(4, 1, 3)
        # plt.plot(time, state,time, phi_sb_sel, time, phi_sc)
        # plt.gca().set_xlim(left=LLIM, right=RLIM)
        # plt.subplot(4, 1, 4)
        # plt.plot(time, np.transpose(vcsb))
        # plt.gca().set_xlim(left=LLIM, right=RLIM)

        return subband


class FilterBank:
    def __init__(self, fc, n_cbb):
        self._fc = fc
        self._n_cbb = n_cbb
        self._filters = []
        for i, (f, n) in enumerate(zip(fc, n_cbb)):
            self._filters.append(FilterChan(f, n))

    def generate_subbands(self, signal, fs):
        fs_resamp = fs if fs > self._fc[-1] * 24 else self._fc[-1] * 24 # resample only if needed
        t = np.arange(len(signal)) / fs
        t_resamp = np.arange(round(len(signal) * (fs_resamp / fs))) / fs_resamp
        signal_resamp = np.interp(t_resamp, t, signal) # might be good to check if other interpolations work better
        # subbands_resamp = np.zeros((signal_resamp.shape[0], len(self._filters)))
        subbands = np.zeros((signal.shape[0], len(self._filters)))
        for i, filt in enumerate(self._filters):
            # subbands_resamp[:, i] = filt.generate_subband(signal_resamp, fs_resamp)
            subbands_resamp = filt.generate_subband(signal_resamp, fs_resamp)
            subbands[:, i] = np.interp(t, t_resamp, subbands_resamp)  # resample back to original fs
        return subbands

    def channel_energies(self, signal, fs):
        subbands = self.generate_subbands(signal, fs)
        energies = np.sum(np.square(subbands),0) / subbands.shape[0]
        return energies


class A2IAFE:
    def __init__(self, fclk=510e3, idx_in=None, ncap=None, ndiv=None):
        if idx_in is None:
            idx_in = list(reversed([31,31,31,31,31,31,31,29,28,27,26,25,23,26,25,24,19,24,17,16,15,14,15,21,10,9,12,6,11,3,4,2]))
        if ncap is None:
            ncap = list(reversed([8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 2]))
        if ndiv is None:
            ndiv = [4, 4, 2, 6, 2, 4, 2, 2, 7, 3, 2, 2, 2, 2, 4, 2, 3, 3, 3, 2, 2, 2, 2, 2, 2, 11, 10, 9, 8, 7, 6, 5, 1]

        self._fclk = fclk
        self._idx_in = idx_in # zero-indexed
        self._ncap = ncap
        self._nsc = 3
        self._npath = 4
        self._ndiv = ndiv
        self._nlcm = np.lcm(self._nsc,self._npath) # clock to each filter needs to be divisible by N and switch cap freq multiple

    # Calculate resulting channel frequencies from implemented values
    def clkgen_freq_calc(self):
        fout = self._fclk * np.ones((len(self._ncap)+1,))
        ntot = np.ones((len(self._ncap)+1,))
        frc = np.zeros((len(self._ncap),))

        # https://stackoverflow.com/questions/529424/traverse-a-list-in-reverse-order-in-python
        for i,nc in reversed(list(enumerate(self._ncap))):
            fout[i] = fout[self._idx_in[i]+1]/self._ndiv[i]
            frc[i] = fout[i]/4/(2*np.pi*nc)
            ntot[i] = ntot[self._idx_in[i]+1] * self._ndiv[i]
        ntot = 12*ntot[:-1]
        fc = fout[:-1]/self._nlcm

        return fc, frc, self._ncap, ntot


def generate_tone(a, fc, ftone, fs, maxcycle=1000):
    """
    a: amplitude
    fc: center frequency of filter
    ftone: input frequency offset from center frequency fc
    fs: sample rate desired
    maxcycle: used to limit the number of total points generated if no integer cycles can be found
    """
    min_cycles = 5
    Tmix = 1 / fc
    Tin = 1 / abs(ftone)
    fin = fc + ftone
    frac = Fraction(min(Tmix, Tin) / max(Tmix, Tin)).limit_denominator(maxcycle)
    ncycles = frac.denominator
    Npoints = round(round(fs * max(ncycles, min_cycles) * max(Tmix, Tin)) / 2) * 2
    x = np.arange(Npoints)
    tin = x / fs
    tone = a * np.sin(2 * np.pi * fin * tin)
    print('Generating envs for ', str(ftone), ' Hz tone using ', str(Npoints), ' points for ', max(ncycles, min_cycles), ' cycles')
    return tin, tone