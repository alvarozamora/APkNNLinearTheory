import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from colossus.cosmology import cosmology
from scipy.special import spherical_jn as spjn
from scipy.interpolate import interp2d
from pqdm.processes import pqdm

def ConeBand(alpha = np.pi/3, N=10**5):
    
    R = np.random.uniform(0,1,N)**(1/3)
    t = np.arccos(np.random.uniform(-1,1, N))
    p = np.random.uniform(0, 2*np.pi, N)

    x = R*np.cos(p)*np.sin(t)
    y = R*np.sin(p)*np.sin(t)
    z = R*np.cos(t)

    cones = (t < alpha) | (t > np.pi-alpha)

    conex = x[cones]; coney = y[cones]; conez = z[cones]; coner = R[cones]
    bandx = x[np.invert(cones)]; bandy = y[np.invert(cones)]; bandz = z[np.invert(cones)]; bandr = R[np.invert(cones)]
    
    return conex, coney, conez, bandx, bandy, bandz


def delta(r, t, p): 
    return r*np.array([np.sin(t)*np.cos(p), np.sin(t)*np.sin(p),np.cos(t)]).reshape(1,3)

def InCone(pos, alpha = np.pi/3):
    
    # Unpack Cartesian
    x, y, z = pos.T
    
    # Compute distance from origin
    r = np.sqrt(x*x + y*y + z*z)
    
    # Compute angle wrt z-axis
    theta = np.arccos(z/r)
    
    # Check if incone
    incone = (theta <= alpha) | (theta >= (np.pi - alpha))
    
    # Also check if in sphere
    alsoinsphere = incone & (r < 1)
    
    return alsoinsphere.mean()

def InBand(pos, alpha = np.pi/3):
    
    # Unpack Cartesian
    x, y, z = pos.T
    
    # Compute distance from origin
    r = np.sqrt(x*x + y*y + z*z)
    
    # Compute angle wrt z-axis
    theta = np.arccos(z/r)
    
    # Check if inband
    inband = (theta >= alpha) & (theta <= (np.pi - alpha))
    
    # Also check if in sphere
    alsoinsphere = inband & (r < 1)
    
    return alsoinsphere.mean()
    

#def fraction(r, t, p, alpha = np.pi/3, N=10**5):
def fraction(args):
    
    r, t, p, alpha, N = args

    from time import time 
    start = time()
    
    conex, coney, conez, bandx, bandy, bandz = ConeBand(alpha, N=N)
    
    newcone = np.array([conex, coney, conez]).T + delta(r,t,p)
    newband = np.array([bandx, bandy, bandz]).T + delta(r,t,p)
    
    results = InCone(newcone, alpha), InBand(newband, alpha)
    
    return results


if __name__ == "__main__":


    ds = np.linspace(0,2,100)
    ts = np.linspace(0,np.pi/2,100)
    args = [(d,t,0,np.pi/3, 10**7) for d in ds for t in ts]
    results = np.array(pqdm(args, fraction, 8))
        

    coneresults, bandresults = results.T
    coneresults = coneresults.reshape(len(ds),len(ts)).T
    bandresults = bandresults.reshape(len(ds),len(ts)).T

    np.save('ConeWeightFunction', coneresults)
    np.save('BandWeightFunction', bandresults)


    bandinterpolator = interp2d(ds, ts, bandresults)
    coneinterpolator = interp2d(ds, ts, coneresults)

    dmesh, tmesh = np.meshgrid(ds, ts)

    # Verify that the interpolating order is the same
    plt.figure(figsize=(16,8))
    plt.subplot(121)
    plt.imshow(coneresults, extent=(ds.min(),ds.max(),ts.min(),ts.max()), aspect='auto')
    plt.subplot(122)
    plt.imshow(coneinterpolator(ds, ts), extent=(ds.min(),ds.max(),ts.min(),ts.max()), aspect='auto')
    plt.savefig('coneweight.png')

    plt.figure(figsize=(16,8))
    plt.subplot(121)
    plt.imshow(bandresults, extent=(ds.min(),ds.max(),ts.min(),ts.max()), aspect='auto')
    plt.subplot(122)
    plt.imshow(bandinterpolator(ds, ts), extent=(ds.min(),ds.max(),ts.min(),ts.max()), aspect='auto')
    plt.savefig('bandweight.png')
