import sc_filterbank
import numpy as np
from fractions import Fraction
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('MacOSX')
matplotlib.interactive(True)
plt.ion()

fs = 4*16e3 # sample rate

afe = sc_filterbank.A2IAFE()
fc, frc, ncbb, ntot = afe.clkgen_freq_calc()
# fc = [1e3,2e3,3e3] # filter center freq
# ncbb = [6,7,8] # N-path caps

chan = 23
ftone = 10
A = 1

tin,input_sound = sc_filterbank.generate_tone(A, fc[chan], ftone, fs, 10)

# scfb = sc_filterbank.FilterBank(fc[chan:chan+2], ncbb[chan:chan+2])
scfb = sc_filterbank.FilterBank(fc, ncbb)
subbands = scfb.generate_subbands(input_sound, fs)

plt.plot(tin, subbands)
plt.show()
