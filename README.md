# sc_filterbank
Switched-capacitor filterbank model for audio feature extraction

**Quickstart usage:**

```python
import sc_filterbank

# instantiate an A2IAFE object
afe = sc_filterbank.A2IAFE()

# for filtered outputs in the same time-base as the input signal use subbands
# input_signal: 1-D numpy array
# fs: sample rate (float)
# subbands = [N,32] numpy array
subbands = afe.subbands(input_signal, fs)

# for average channel energies for a given input
# energies = [,32] numpy array
energies = afe.energies(input_signal, fs)
```
Additional usage examples for customizing channel frequencies, bandwidths and number of channels are found in
```python
tb_filterbank.py
tb_channel.py
```
