import numpy as np
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology
from colossus.lss import peaks, bias

# Define Quijote Cosmology
Om0 = 0.3175
quijote = {'flat': True, 'H0': 67.11, 'Om0': Om0, 'Ob0': 0.049, 'sigma8': 0.834, 'ns': 0.9624, }
cosmology.addCosmology('Quijote', quijote)
quijote = cosmology.setCosmology('Quijote')


def AverageBias(masses, z):

    return bias.haloBias(masses, model = 'tinker10', z = z, mdef = 'vir').mean()




if __name__ == "__main__":

    #### Theoretical Calculation: Bias for a mass grid ###
    # Define mass grid and redshift
    M = np.logspace(12, 15, 1000)
    z = 0.5

    # Convert mass to peak height
    nu = peaks.peakHeight(M, z)

    # Compute bias for two models
    b1 = bias.haloBiasFromNu(nu, model = 'sheth01')
    b2 = bias.haloBias(M, model = 'tinker10', z = z, mdef = 'vir')

    #### Numerical Caluclation: Compute average bias for these two models  ####
    # Load in quijote box (pre-filtered for most massive 1e5 halos)
    masses = np.load('OneBox.npz')['mass']
    nb1 = bias.haloBiasFromNu(peaks.peakHeight(masses, z), model = 'sheth01').mean()
    nb2 = bias.haloBias(masses, model = 'tinker10', z = z, mdef = 'vir').mean()
    print(f'nb1 = {nb1}, nb2 = {nb2}')

    print(np.log10(masses.mean()), np.log10(masses.min()), np.log10(masses.max()))

    plt.figure(figsize=(8,12))
    plt.subplot(211)
    plt.semilogx(M, b1, 'C0', label='sheth01')
    plt.semilogx(M, b2, 'C1', label='tinker10')
    plt.semilogx(M, nb1*np.ones_like(M), 'C0--', label='1e5 quijote mean sheth01')
    plt.semilogx(M, nb2*np.ones_like(M), 'C1--', label='1e5 quijote mean tinker01')
    plt.legend()
    plt.xlabel('Mass')
    plt.ylabel(r'$b$')

    plt.subplot(212)
    plt.hist(masses, bins=np.logspace(12, 15, 100))
    plt.xlim(M.min(),M.max())
    plt.gca().set_xscale("log")
    plt.savefig('bias.png')
