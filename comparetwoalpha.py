import numpy as np; sq = np.sqrt
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree as Tree
from utils import equalsky

j = 2
subsamples = S = 10**j

# nbar = 1e5/1e9
#onereal = np.load('pcdf_0_0.npz')
#onered  = np.load('pcdf_2_0.npz')
#onereal = np.load('pcdf_0_0_n=5_z=1.00_alpha.npz')
#onered  = np.load('pcdf_2_0_n=5_z=1.00_alpha.npz')

# nbar = 1e4/1e9
#onereal = np.load('pcdf_0_0_n=4_z=1.00_alpha.npz')
#onered  = np.load('pcdf_2_0_n=4_z=1.00_alpha.npz')

# nbar = 1e3/1e9
onereal = np.load('pcdf_0_0_n=3_z=1.00_alpha.npz')
onered  = np.load('pcdf_2_0_n=3_z=1.00_alpha.npz')


fig, axes = plt.subplots(2,2,figsize=(12,8), gridspec_kw={'height_ratios': [4, 1]})

# Theoretical Predictions
axes[0][0].loglog(onereal['R'], onereal['pcdf'], label='real')
axes[0][0].loglog(onered ['R'], onered ['pcdf'], label='red')
axes[0][0].set_xlabel(r'Distance $h^{-1}$ Mpc')
axes[0][0].set_title('Theoretical Peaked 1NN-CDFs')
axes[0][0].set_ylabel(r'Peaked CDF')
axes[0][0].legend()
axes[0][0].set_ylim(1e-2, 1)
axes[0][0].set_xlim(onereal['R'].min(),onereal['R'].max())
# Theoretical Residuals
axes[1][0].semilogx(onered['R'], onered['cdf']-onereal['cdf'])
axes[1][0].set_xlabel(r'Distance $h^{-1}$ Mpc')
axes[1][0].set_ylabel(r'Residual')
axes[1][0].set_xlim(onereal['R'].min(),onereal['R'].max())
#axes[1][0].set_ylim(-0.015, 0.01)
axes[1][0].set_ylim(-0.005, 0.001)

# Simulation
data = np.load('OneBox.npz') # Redshift = 0.5
data = np.load('OneBox_1.npz') # Redshift = 1.0
rpos = data['rpos']; np.random.shuffle(rpos); rpos = np.split(rpos, S)
pos  = data['pos' ]; np.random.shuffle( pos);  pos = np.split( pos, S)
rr = []; sr = []
Nrand = 10**5
for i in range(S):
   rtree = Tree(rpos[i], leafsize=16, compact_nodes=True, balanced_tree=True, boxsize=1000)
   stree = Tree( pos[i], leafsize=16, compact_nodes=True, balanced_tree=True, boxsize=1000)
   query = np.random.uniform(size=(Nrand,3))*1000
   rr_, rids = rtree.query(query, k=16); 
   sr_, sids = stree.query(query, k=16); 

   _, rr_ = equalsky(rpos[i], query, rr_, rids, k=1, bs=1000, verb=True) # discard perpendicular (band), keep cone
   _, sr_ = equalsky( pos[i], query, sr_, rids, k=1, bs=1000, verb=True) # 

   rr.append(rr_)
   sr.append(sr_)
reduce = rd = 100
rr = np.sort(np.concatenate(rr))[::rd]
sr = np.sort(np.concatenate(sr))[::rd]
cdf = np.arange(1,S*Nrand+1)/(S*Nrand); cdf = cdf[::rd]; pcdf = np.minimum(cdf, 1-cdf)

# Load avg CDFs
data = np.load('final_cdf.npz')
avg = data['avg']; pavg = data['pavg']; err = data['err']
R = np.logspace(np.log10(2),np.log10(35),100)*10**(j/3)
axes[0][1].loglog(R, pavg[j,0], label='real')
axes[0][1].loglog(R, pavg[j,1], label='red')
axes[0][1].set_xlabel(r'Distance $h^{-1}$ Mpc')
axes[0][1].set_title('Measured Peaked 1NN-CDFs')
axes[0][1].set_ylabel(r'Peaked CDF')
axes[0][1].legend()

axes[0][1].set_ylim(1e-2, 1)
axes[0][1].set_xlim(onereal['R'].min(),onereal['R'].max())

axes[1][1].semilogx(R, avg[j,1] - avg[j,0])
axes[1][1].set_xlabel(r'Distance $h^{-1}$ Mpc')
axes[1][1].set_ylabel(r'Residual')
axes[1][1].set_xlim(onereal['R'].min(),onereal['R'].max())
#axes[1][1].set_ylim(-0.015, 0.01)
axes[1][1].set_ylim(-0.005, 0.001)



plt.savefig(f'comparison_z=1_S={S}.png', dpi=230)


# Both on Same

plt.figure(figsize=(12,8))
plt.loglog(onereal['R'], onereal['pcdf'], label='Treal', linewidth=2)
plt.loglog(onered ['R'], onered ['pcdf'], label='Tred', linewidth=2)
plt.errorbar(R, pavg[j,0], yerr=err[j,0], ls='--', label='Nreal')
plt.errorbar(R, pavg[j,1], yerr=err[j,1], ls='--', label='Nred')
#plt.loglog(R, pavg[j,0], label='Nreal')
plt.xlabel(r'Distance $h^{-1}$ Mpc')
plt.title('Measured Peaked 1NN-CDFs')
plt.ylabel(r'Peaked CDF')
plt.xlim(onereal['R'].min(), onereal['R'].max())
plt.ylim(1e-2)
plt.legend()

plt.savefig(f'combined_comparison_z=1_S={S}.png', dpi=1000)