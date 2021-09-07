import numpy as np
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology
from scipy.integrate import tplquad, dblquad, quad, simps
from scipy.integrate import nquad
from scipy.special import jv
import yt; yt.enable_parallelism(); print(yt.is_root())


# Custom Plot Formatting
plt.rcParams['figure.facecolor'] ='white'
plt.rcParams['figure.figsize']   = (12,8)
plt.rcParams['xtick.labelsize']  = 15
plt.rcParams['ytick.labelsize']  = 15
plt.rcParams['axes.titlesize']  = 15
plt.rcParams['axes.labelsize']  = 15

# Define Quijote Cosmology
Om0 = 0.3175
quijote = {'flat': True, 'H0': 67.11, 'Om0': 0.3175, 'Ob0': 0.049, 'sigma8': 0.834, 'ns': 0.9624, }
cosmology.addCosmology('Quijote', quijote)
quijote = cosmology.setCosmology('Quijote')

def mu(k,t=0):
    return np.cos(t)

def Psv(k, t, z=1, sv=200, b=1):
    return quijote.matterPowerSpectrum(k=k,z=z)*(b + Om0**0.6 * mu(k,t)**2)**2/(1 + (k*mu(k,t)*sv)**2/2)**2

r = 10.
rmu = 0.
z = 1.
sv = 0.
b = 1.

#def innerintegrand(t,k,r,rmu,z,sv,b):
    
#def integrand(k,r,rmu,z,sv,b):

def integrand(k,t,r=r,rmu=rmu,z=z,sv=sv,b=b):
    
    #unpack arguments
    #k, t, r, mu, z, sv, b = args
    #print(k,t,r,rmu,z,sv,b)
    
    # argument of bessel
    absb = np.abs(k*r*np.sin(t)*np.sqrt(1-rmu*rmu))
    
    return k**2 * np.sin(t) * Psv(k, t, z=z, sv=sv, b=b) * np.exp(-1j*k*r*rmu*np.cos(t)) * 2*np.pi*jv(0,absb)
    #return k**2 * Psv(k, t, z=z, sv=sv, b=b) * np.exp(-1j*k*r*rmu*np.cos(t)) * 2*np.pi*jv(0,absb)
    
#integrand = lambda kk, tt: kk**2 * Psv(kk, tt, z=z, sv=sv, b=b) * np.exp(-1j*kk*r*rmu*np.cos(tt)) * 2*np.pi*jv(0,np.abs(kk*r*np.sin(tt)*np.sqrt(1-rmu*rmu)))

kmin = 1e-3
#print(dblquad(integrand, 0, np.pi, 0, np.inf, args=(r,rmu,z,sv,b)))
#print(dblquad(integrand, 0, np.pi, kmin, np.inf, args=args))

k = np.logspace(-3,3,10000)
'''
plt.figure()
plt.loglog(k, quijote.matterPowerSpectrum(k,z=z))
plt.loglog(k, Psv(k,np.pi/2,z=z,sv=sv,b=b),'--')
plt.loglog(k, Psv(k,0,z=z,sv=sv,b=b),'--')
'''
if yt.is_root():
    plt.figure()
    plt.semilogx(k, integrand(k,0))
    plt.semilogx(k, integrand(k,np.pi/2))
    #plt.loglog(k, integrand(k,np.pi/2))
    #plt.loglog(k, integrand(k,np.pi/2,r,rmu,z,sv,b),'--')
    absb = np.abs(k*r*np.sin(0)*np.sqrt(1-rmu*rmu))
    expfactor = np.exp(-1j*k*r*rmu*np.cos(np.pi/2))
    expfactor = np.exp(-1j*k*r*rmu*np.cos(0))
    #print(expfactor)
    #plt.loglog(k, k**2*quijote.matterPowerSpectrum(k,z=z)*2*np.pi*jv(0,absb)*2*np.pi*expfactor)
    t = np.pi/2
    absb = np.abs(k*r*np.sin(t)*np.sqrt(1-rmu*rmu))
    #plt.semilogx(k, k**2 * Psv(k, t, z=z, sv=sv, b=b) * np.exp(-1j*k*r*rmu*np.cos(t)) * 2*np.pi*jv(0,absb),'k--')
    t = 0
    absb = np.abs(k*r*np.sin(t)*np.sqrt(1-rmu*rmu))
    #plt.semilogx(k, k**2 * Psv(k, t, z=z, sv=sv, b=b) * np.exp(-1j*k*r*rmu*np.cos(t)) * 2*np.pi*jv(0,absb),'k--')
    #plt.show()


xf = 6
x = np.arange(xf)
y = np.arange(xf)



storage = {}
for sto, n in yt.parallel_objects(range(xf*xf), 0, storage=storage):
        
        i = n//xf
        j = n%xf

        # compute coordinate
        r = np.sqrt(x[i]*x[i] + y[j]*y[j])
        if r == 0:
            rmu = 0
        else:
            rmu = y[j]/r

        options={'limit':1000}
        #result = dblquad(integrand, 0, np.pi, 1e-3, 1e6, args=(r, rmu, z, sv, b))
        result = nquad(integrand,[[1e-3,1e2],[0,np.pi]],
            args=(r,rmu,z,sv,b),opts=[options,options])
        print(i, j, r, rmu, result[0], result[1], result[1]/result[0])

        sto.result = result[0]
        sto.result_id = f'{i}_{j}'

if yt.is_root():

    Xis = np.zeros((xf,xf)) 
    for i in range(xf):
        for j in range(xf):
            Xis[i,j] = storage[f"{i}_{j}"]

    plt.figure()
    plt.imshow(Xis, origin='lower', extent=(x.min(),x.max(),y.min(),y.max()))
    plt.xlabel(r'Perpendicular Distance ($h^{-1}$ Mpc)')
    plt.ylabel(r'Line-of-Sight Distance ($h^{-1}$ Mpc)')
    plt.colorbar()
    plt.show()


        