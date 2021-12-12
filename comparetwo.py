import numpy as np
import matplotlib.pyplot as plt


onereal = np.load('pcdf_0_0.npz')
onered  = np.load('pcdf_2_0.npz')

fig, axes = plt.subplots(2,1,figsize=(12,20))
axes[0].loglog(onereal['R'], onereal['pcdf'], label='real')
axes[0].loglog(onered ['R'], onered ['pcdf'], label='red')
axes[0].set_xlabel(r'Distance $h^{-1}$ Mpc')
axes[0].set_title('Peaked 1NN-CDFs')
axes[0].set_ylabel(r'Peaked CDF')
axes[0].legend()
axes[0].set_ylim(1e-2)

axes[1].semilogx(onereal['R'], np.abs(onereal['pcdf']-onered['pcdf']))
axes[1].set_xlabel(r'Distance $h^{-1}$ Mpc')
axes[1].set_title('Peaked 1NN-CDFs')
axes[1].set_ylabel(r'Peaked CDF')


plt.savefig('comparison.png', dpi=230)

