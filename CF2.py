import numpy as np
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology
from scipy.integrate import tplquad, dblquad, quad, simps
from scipy.integrate import nquad
from scipy.special import jv, jn_zeros
import yt; yt.enable_parallelism(); print(yt.is_root()); is_root = yt.is_root(); from yt.config import ytcfg; cfg_option = "__topcomm_parallel_rank"; cpu_id = ytcfg.getint("yt", cfg_option)
from mpmath import quadosc
from scipy.special import spherical_jn as spjn
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--cg', type=int, default=5)
parser.add_argument('-c', type=int, default=1)
args = parser.parse_args()

# Custom Plot Formatting
plt.rcParams['figure.facecolor'] ='white'
plt.rcParams['figure.figsize']   = (12,8)
plt.rcParams['xtick.labelsize']  = 15
plt.rcParams['ytick.labelsize']  = 15
plt.rcParams['axes.titlesize']  = 15
plt.rcParams['axes.labelsize']  = 15

# Define Quijote Cosmology
Om0 = 0.3175
quijote = {'flat': True, 'H0': 67.11, 'Om0': Om0, 'Ob0': 0.049, 'sigma8': 0.834, 'ns': 0.9624, }
cosmology.addCosmology('Quijote', quijote)
quijote = cosmology.setCosmology('Quijote')

def mu(k,t=0):
    return np.cos(t)

def Psv(k, t, z=1, sv=200, b=1):
    return quijote.matterPowerSpectrum(k=k,z=z)*(b + Om0**0.6 * mu(k,t)**2)**2/(1 + (k*mu(k,t)*sv)**2/2)**2

def TwoEllipsoidCaps(d, t, s, R):

    dmax = (2*np.sqrt(2)*R*s)/np.sqrt(1 + s**3 + np.cos(2*t) - s**3*np.cos(2*t))

    if d < dmax:
        return (np.pi*(64*R**3 - (24*np.sqrt(2)*d*R**2*np.sqrt(1 + s**3 - (-1 + s**3)*np.cos(2*t)))/s + (np.sqrt(2)*d**3*(1 + s**3 - (-1 + s**3)*np.cos(2*t))**1.5)/s**3))/48.
    else:
        return 0 

def DoubleSphereVolume(R,d):
    if d < 2*R:
        return np.pi/12 * (d - 2*R)**2 * (d + 4*R)
    else:
        return 0

z = 1.0
sv = 0
b = 3.3 # * 0.875
#b = 3.348305028707007 * 1.01902463
#b = 3.348305028707007 * 1.75
f = Om0**0.6

def integrand(t,r,rmu,z=z,sv=sv,b=b):
    
    # argument of bessel
    absb = lambda k: np.abs(k*r*np.sin(t)*np.sqrt(1-rmu*rmu))

    def first_integrand(k):
        k = np.float64(k)
        if k < 0:
            print(k)
        return k**2 * np.sin(t) * Psv(k, t, z=z, sv=sv, b=b) * np.cos(    k*r*rmu*np.cos(t)) * 2*np.pi*jv(0,absb(k))

    # Find Zeros Function
    kp1 = r*rmu*np.cos(t)
    kp2 = r*np.sqrt(1-rmu*rmu)*np.sin(t)
    Nz = 1000
    coszeros = (np.pi*np.arange(1,Nz+1) + np.pi/2)/kp1
    j0zeros = jn_zeros(0,Nz)/kp2
    zeros = np.sort(np.append(coszeros, j0zeros))
    # Remove zeros and nans
    zeros = zeros[np.invert(np.isnan(zeros))]
    zeros = zeros[zeros<np.inf]
    zeros = zeros[zeros>0]

    def z0s(n):
        assert np.abs(n-int(n))<1e-6,"something is wrong"
        return zeros[int(n)]
    
    period = np.array([np.diff(np.sort(coszeros)).mean(), np.diff(np.sort(j0zeros)).mean()])
    period = np.min(period[np.invert(np.isnan(period))])
    zeros = lambda n: n*period/2
    #if is_root:
        #print(f"integrating theta={t:.4f}")

    n = 0
    eps = 100
    result = 0
    while eps > 1e-6:

        options = {'limit': 1000}

        res = 0
        res_ = nquad(first_integrand, [[zeros(n), zeros(n+1)]], opts=options)
        assert np.abs(res_[1]/res_[0]) < 1e-4, f"too much error {res_[1]/res_[0]:.3e}, n={n}"
        res += res_[0]
        n += 1

        res_ = nquad(first_integrand, [[zeros(n), zeros(n+1)]], opts=options)
        assert np.abs(res_[1]/res_[0]) < 1e-4, f"too much error {res_[1]/res_[0]:.3e}, n={n}"
        res += res_[0]
        n += 1

        result += res
        eps = res/result
        #if is_root:
        #    print(n, eps)



    #result = np.float64(quadosc(first_integrand, [1e-4, np.inf], zeros = zeros))
    #print(f"result is = {result:.3f} for r = {r:.3f}, rmu = {rmu:.3f}")
    return result


xf = 10
L = 40
x = L*np.arange(xf)/xf+1/2
y = L*np.arange(xf)/xf+1/2


storage = {}
def Xi(r, rmu, z=z, sv=sv, b=b):

    # This part could work
    #options={'limit':50}
    #result = nquad(integrand, [[0, np.pi]], args=(r,rmu,z,sv,b), opts=options)
    #assert np.abs(result[1]/result[0]) < 1e-3, "Too Much Error"
    #return result[0]

    # Newton-Cotes 4th Order
    A = 1e-6; B = np.pi-1e-6 # Bounds
    N = 4*50+1 # Needs to be 4k+1
    n = N//4 # number of stencils
    h = (B-A)/(N-1)*4 # step size
    w = np.zeros(N) # weights 
    x = A + np.arange(N)*h/4 #grid position
    for i in range(n):
        w[4*i:4*i+5] += np.array([7, 32, 12, 32, 7]) * h/90

    results = {}
    for sto, X in yt.parallel_objects(x, args.c, dynamic=False, storage=results):

        j = np.where(x==X)[0][0]

        sto.result = integrand(X, r=r,rmu=rmu)
        sto.result_id = f'{j}'
    
    f = np.array([results[f"{i}"] for i in range(N)])
    #f = np.array([integrand(q, r=r,rmu=rmu) for q in x])
    #print(f)
    #print(val)
    if is_root:
        print((w*f).sum())
    return (w*f).sum()


# Donghui Jeong Liang Dai, Marc Kamionkowski, and Alexander S. Szalay 2014 (Eq 2, 3)
def XiRSDApprox(r, rmu, z=z, sv=sv, b=b, separateterms=False):

    # Define Legendre Polynomials
    P0 = lambda x: 1
    P2 = lambda x: ( 3*x**2 - 1)/2
    P4 = lambda x: (35*x**4 - 30*x**2 + 3)/8

    # Compute Other Moments
    # First: Power Spectrum
    Pk = lambda k: quijote.matterPowerSpectrum(k=k, z=z)
    def xin(r, n):
        
        if n!=0 or True:

            integrand = lambda k: k**2 * Pk(k) / (2 * np.pi**2) * spjn(n, k*r)  # TODO: Check Normalization
            
            # integrate function
            #result = nquad(integrand, [[1e-10, 1e10]],opts={'limit': 1000})
            #assert np.abs(result[1]/result[0]) < 1e-3, f"Too much integ error ({np.abs(result[1]/result[0]):.3e})"
            #print(f"xin integ error is {np.abs(result[1]/result[0]):.3e}")
            #return result[0]

            # integrate log-grid
            # smaller r's result in larger periods for k --> need wider grid
            # r in Mpc/h
            Npts = 10000
            kgrid = np.sort(np.concatenate([np.logspace(-6,0,Npts), np.linspace(1 + 1/Npts, np.maximum(50/r,20), Npts)]))
            integrand_grid = integrand(kgrid)
            result = simps(integrand_grid, kgrid)
            assert not np.isnan(result), "xin result is nan"
            return result
            
        else:
            return quijote.correlationFunction(r,z)

    if not separateterms:
        result = (b*b + 2/3*b*f + f*f/5)*xin(r, 0)*P0(rmu) - (4/3*b*f + 4/7*f*f)*xin(r, 2)*P2(rmu) + 8/35*f*f*xin(r, 4)*P4(rmu)

        return result
    else:
        
        # Only Compute Power Spectrum Moments Once
        xi0 = xin(r, 0)
        xi2 = xin(r, 2)
        xi4 = xin(r, 4)

        # Squared, Linear, and Constant terms (with respect to bias parameter)
        b2term = b*b*(xi0*P0(rmu))
        b1term = b*(2/3*f*xi0*P0(rmu) + 4/3*f*xi2*P2(rmu))
        b0term = f*f/5*xi0*P0(rmu) + 4/7*f*f*xi2*P2(rmu) + 8/35*f*f*xi4*P4(rmu)

        return b2term, b1term, b0term
    

        


def XiIntegral(R, s, R1=1e-3, RSD=2, T = [0, np.pi]):
    
    # We add a 2*pi here from the azimuthal integral that would need to be done in the following lines
    weight = lambda r, t: 2*np.pi*TwoEllipsoidCaps(r, t, s, R)

    if RSD == 1:
        if is_root:
            print(RSD)
        integrand = lambda r, t: weight(r,t)*Xi(r,np.cos(t)) * r*r*np.sin(t)
    elif RSD == 2:
        if is_root:
            print("Using Dipole + Quadruple O(f^2) Approximation")
        integrand = lambda r, t: weight(r,t)*XiRSDApprox(r,np.cos(t)) * r*r*np.sin(t)
    elif RSD == 0:
        integrand = lambda r, t: weight(r,t)*b*b*quijote.correlationFunction(r,z) * r*r*np.sin(t)
    elif RSD not in [0,1,2]:
        assert False, "invalid RSD mode"

    dlims = lambda t: [R1, (2*np.sqrt(2)*R*s)/np.sqrt(1 + s**3 + np.cos(2*t) - s**3*np.cos(2*t))]

    #result = nquad(integrand, [dlims, T],opts={'limit': 1000}) # 2D Integral

    # 1D Integral, other done w/ simps
    def new_integrand(t): 
        lims = dlims(t)
        rgrid = np.linspace(lims[0], lims[1], 1000) # Linear Grid
        #rgrid = np.logspace(np.log10(lims[0]), np.log10(lims[1]), 1000) # Logarithmic Grid
        result = simps([integrand(rg, t) for rg in rgrid], rgrid)
        assert not np.isnan(result), "new_integrand result is nan"
        return result
    
    # Functional integral
    #result = quad(new_integrand, *T) 
    #print(f"Xi({R:.2f}) Integral Error is {np.abs(result[1]/result[0]):.3e}")
    #return result[0]

    # Simpsons Rule
    thgrid = np.linspace(0, np.pi, 200)
    result = simps([new_integrand(th) for th in thgrid], thgrid)
    return result
    

    




S = [1.0]#, 0.98, 0.99, 1.01, 1.02]
n = 3
R = np.logspace(np.log10(2),np.log10(35),100)*10**((5-n)/3)
RSD = 0
results = []

params = [(s, r) for s in S for r in R]

xis = {}
for sto, (s, r) in yt.parallel_objects(params, args.cg, dynamic=True, storage=xis):

    j = params.index((s,r))

    print(f"Rank {cpu_id} is working on item {j}, which is r={r:.3f} Mpc/h for s={s:.3f}")

    sto.result = XiIntegral(r, s, RSD=RSD)
    sto.result_id = f"{j}"

    print(f"Rank {cpu_id} finished")


CDFs = np.empty((len(R),len(S))); pCDFs = np.empty((len(R),len(S)))
twoCDFs = np.empty((len(R),len(S))); twopCDFs = np.empty((len(R),len(S)))

if is_root:
    for s in range(len(S)):

        #results = np.array([xis[f"{len(R)*s + j}"] for j in range(len(R))])
        results = np.array([xis[len(R)*s + j] for j in range(len(R))])

        nbar = 10**n/1e9

        onenn = 1 - np.exp(-nbar*4*np.pi*R**3/3 + nbar*nbar/2*results)

        mnV = -nbar*4*np.pi*R**3/3
        nbsqxiover2 = nbar*nbar/2*results
        if s==0:
            np.savez(f'terms_{RSD}_z={z:.2f}_n={n}', mnV=mnV, nbsqxiover2=nbsqxiover2)
        onennpcdf = np.minimum(onenn, 1-onenn)

        twonn = onenn - np.exp(-nbar*4*np.pi*R**3/3 + nbar*nbar/2*results) * (nbar*4*np.pi*R**3/3 - nbar*nbar*results)
        twonnpcdf = np.minimum(twonn, 1-twonn)

        CDFs[:,s]= onenn; pCDFs[:,s]= onennpcdf; twoCDFs[:,s] = twonn; twopCDFs[:, s] = twonnpcdf; 

CDFs = CDFs.T; pCDFs = pCDFs.T; twoCDFs = twoCDFs.T; twopCDFs = twopCDFs.T;


if is_root:

    fig, ax = plt.subplots(2,2,figsize=(20,16))

    for q, pcdf in enumerate(pCDFs):
        ax[0][0].loglog(R, pcdf, '.-', label=f"{S[q]:.2f}")
        np.savez(f'pcdf_{RSD}_{q}_n={n}_z={z:.2f}',R=R,pcdf=pcdf, cdf=CDFs[q])
    ax[0][0].set_ylim(1e-3)
    ax[0][0].set_xlabel(r'Distance $h^{-1}$ Mpc')
    ax[0][0].set_title('Peaked 1NN-CDFs')
    ax[0][0].set_ylabel(r'Peaked CDF')
    ax[0][0].legend()

    for q, cdf in enumerate(CDFs[1:],1):
        ax[0][1].semilogx(R, cdf-CDFs[0],f'C{q}.-', label=f"{S[q]:.2f}")
    ax[0][1].set_title('1NN Residual')
    ax[0][1].set_xlabel(r'Distance $h^{-1}$ Mpc')
    ax[0][1].set_ylabel(r'Peaked CDF Residual')
    ax[0][1].legend()

    for q, pcdf in enumerate(twopCDFs):
        ax[1][0].loglog(R, pcdf, '.-', label=f"{S[q]:.2f}")
    ax[1][0].set_ylim(1e-3)
    ax[1][0].set_xlabel(r'Distance $h^{-1}$ Mpc')
    ax[1][0].set_title('Peaked 2NN-CDFs')
    ax[1][0].set_ylabel(r'Peaked CDF')
    ax[1][0].legend()

    for q, cdf in enumerate(twoCDFs[1:],1):
        ax[1][1].semilogx(R, cdf-twoCDFs[0],f'C{q}.-', label=f"{S[q]:.2f}")
    ax[1][1].set_title('2NN Residual')
    ax[1][1].set_xlabel(r'Distance $h^{-1}$ Mpc')
    ax[1][1].set_ylabel(r'Peaked CDF Residual')
    ax[1][1].legend()


    plt.savefig(f'onennpcdf_{RSD}_z={z:.2f}.png')



'''
# This block tests Xi and produces a 2D plot
for sto, n in yt.parallel_objects(range(xf*xf)[::-1], 0, storage=storage, dynamic=True):
        
        i = n//xf
        j = n%xf

        # compute coordinate
        r = np.sqrt(x[i]*x[i] + y[j]*y[j])
        if r == 0:
            rmu = 0
        else:
            rmu = y[j]/r

        print(r, rmu)
        sto.result = Xi(r, rmu)
        print(f"{i}_{j}")
        sto.result_id = f"{i}_{j}"



if yt.is_root():

    Xis = np.zeros((xf,xf)) 
    for i in range(xf):
        for j in range(xf):
            Xis[i,j] = storage[f"{i}_{j}"]

    plt.figure()
    plt.imshow(np.log(1+Xis), origin='lower', extent=(x.min()-0.5,x.max()+0.5,y.min()-0.5,y.max()+0.5))
    #plt.xticks(np.arange(xf+1), np.arange(xf+1))
    #plt.yticks(np.arange(xf+1), np.arange(xf+1))
    #plt.xticks(L*np.arange(xf+1)/xf, L*np.arange(xf+1)/xf)
    #plt.yticks(L*np.arange(xf+1)/xf, L*np.arange(xf+1)/xf)
    plt.xlabel(r'Perpendicular Distance ($h^{-1}$ Mpc)')
    plt.ylabel(r'Line-of-Sight Distance ($h^{-1}$ Mpc)')
    plt.colorbar()
    plt.savefig('CorrFunc.png',dpi=230)
'''
