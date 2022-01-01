import numpy as np
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology
from colossus.lss import peaks, bias

# Define Quijote Cosmology
Om0 = 0.3175
quijote = {'flat': True, 'H0': 67.11, 'Om0': Om0, 'Ob0': 0.049, 'sigma8': 0.834, 'ns': 0.9624, }
cosmology.addCosmology('Quijote', quijote)
quijote = cosmology.setCosmology('Quijote')


if __name__ == "__main__":

    #### Theoretical Calculation: Bias for a mass grid ###
    # Define mass grid and redshift
    M = np.logspace(12, 15, 1000)
    z = 1.0

    # Convert mass to peak height
    nu = peaks.peakHeight(M, z)

    # Compute bias for two models
    b1 = bias.haloBiasFromNu(nu, model = 'sheth01')
    b2 = bias.haloBias(M, model = 'tinker10', z = z, mdef = 'vir')

    #### Numerical Caluclation: Compute average bias for these two models  ####
    # Load in quijote box (pre-filtered for most massive 1e5 halos)
    # masses = np.load('OneBox.npz')['mass'] # Redshift = 0.5
    masses = np.load('OneBox_1.npz')['mass'] # Redshift = 1
    nb1 = bias.haloBiasFromNu(peaks.peakHeight(masses, z), model = 'sheth01')
    nb2 = bias.haloBias(masses, model = 'tinker10', z = z, mdef = 'vir')
    print(f'nb1 = {nb1.mean()}, nb2 = {nb2.mean()}')

    print(np.log10(masses.mean()), np.log10(masses.min()), np.log10(masses.max()))

    plt.figure(figsize=(8,12))
    plt.subplot(211)
    plt.semilogx(M, b1, 'C0', label='sheth01')
    plt.semilogx(M, b2, 'C1', label='tinker10')
    plt.semilogx(M, nb1.mean()*np.ones_like(M), 'C0--', label='1e5 quijote mean sheth01')
    plt.semilogx(M, nb2.mean()*np.ones_like(M), 'C1--', label='1e5 quijote mean tinker01')
    plt.legend()
    plt.xlabel('Mass')
    plt.ylabel(r'$b$')

    plt.subplot(212)
    plt.hist(masses, bins=np.logspace(12, 15, 100))
    plt.xlim(M.min(),M.max())
    plt.gca().set_xscale("log")
    plt.savefig('bias.png')



    # Subsampling Bias (Deterministic)
    np.random.seed(0)
    nb2 = np.random.permutation(nb2)
    subb1s_10  = np.average(np.array(np.split(nb2, 10)), axis=1)
    subb1s_100 = np.average(np.array(np.split(nb2,100)), axis=1)
    # Ensure Determinism
    # Should print [3.3525 3.3466 3.3409 3.3558 3.3422]
    np.set_printoptions(precision=4)
    print(subb1s_10[:5])

    plt.figure()
    plt.plot([nb2.mean(),nb2.mean()],[0,60],'k--', linewidth=1, alpha=0.7,label='1')
    plt.hist(subb1s_10, bins=np.arange(3,4,0.01), density=True,label='10',alpha=0.3)
    plt.hist(subb1s_100, bins=np.arange(3,4,0.01), density=True,label='100',alpha=0.3)
    plt.legend(title='Subsamples')
    plt.savefig("biases.png", dpi=230)

    print(f"Ratio of 10/100 stdevs  = {subb1s_100.std()/subb1s_10.std():.2f}")


