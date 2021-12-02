import numpy as np
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology
from scipy.integrate import tplquad, dblquad, quad, simps
from scipy.integrate import nquad
from scipy.special import jv, jn_zeros
import yt; yt.enable_parallelism(); print(yt.is_root()); is_root = yt.is_root()
from mpmath import quadosc


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

z = 1.
sv = 0.
b = 1.

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
    while eps > 1e-5:

        options = {'limit': 1000}

        res = 0
        res_ = nquad(first_integrand, [[zeros(n), zeros(n+1)]], opts=options)
        assert np.abs(res_[1]/res_[0]) < 1e-3, f"too much error {res_[1]/res_[0]:.3e}"
        res += res_[0]
        n += 1

        res_ = nquad(first_integrand, [[zeros(n), zeros(n+1)]], opts=options)
        assert np.abs(res_[1]/res_[0]) < 1e-3, f"too much error {res_[1]/res_[0]:.3e}"
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
    f = np.array([integrand(q, r=r,rmu=rmu) for q in x])
    #print(f)
    #print(val)
    print((w*f).sum())
    return (w*f).sum()

def XiIntegral(R, s, R1=0):
    
    # The weight needs a 2pi from the azimuthal integral
    weight = lambda r, t: 2*np.pi*TwoEllipsoidCaps(r, t, s, R)

    integrand = lambda r, t: weight(r,t)*Xi(r,np.cos(t))*r*np.sin(t)

    dlims = lambda t: [R1, (2*np.sqrt(2)*R*s)/np.sqrt(1 + s**3 + np.cos(2*t) - s**3*np.cos(2*t))]

    return nquad(integrand, [dlims, [0, np.pi]])[0]

print(XiIntegral(50,1.2))
assert False
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
        