import numpy as np
from time import time
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology
from scipy.integrate import tplquad, dblquad, quad, simps
#from scipy.integrate import nquad
#from scipy.special import jv, jn_zeros
import yt; yt.enable_parallelism(); print(yt.is_root()); is_root = yt.is_root(); from yt.config import ytcfg; cfg_option = "__topcomm_parallel_rank"; cpu_id = ytcfg.getint("yt", cfg_option)
from scipy.special import spherical_jn as spjn
import argparse
from scipy.interpolate import interpn, interp2d
from scipy.optimize import minimize
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

debug = False
debug_factor = 10 if debug else 1



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
    

#coneweightfunction = np.load('WeightFunction.npy')
coneweightfunction = np.load('ConeWeightFunction.npy')
def ConeWeightFunction(r, t, s, alpha=np.pi/3):

    # Apply Coordinate Transformation due to stretch
    r = r*np.sqrt(1 + s**3 - (-1 + s**3)*np.cos(2*t))/(np.sqrt(2)*s)

    assert (t >= 0) & (t <= np.pi), "Invalid Polar Angle"
    t = np.pi/2 - np.abs(t-np.pi/2)

    # These were the grids with which the array was generated
    ds = np.linspace(0,2,100)
    ts = np.linspace(0,np.pi/2,100) # only goes to pi/2 with new interpolator, but old one goes to pi
    #alphas = np.linspace(0, np.pi/2, 100)
    #x = (alpha, t, r)
    x = (r, t)
    if r < 2:
        #fraction = interpn((alphas, ts, ds), coneweightfunction, x)[0]  # fraction of points still in cone
        fraction = interp2d(ds, ts, coneweightfunction)(*x)[0]
        doubleconevolume = 4 * np.pi / 3 * (1 - np.cos(alpha))          # dimensionless, i.e. R = 1
        return fraction * doubleconevolume
    else:
        return 0 

#bandweightfunction = np.load('WeightFunctionBand.npy')
bandweightfunction = np.load('BandWeightFunction.npy')
def BandWeightFunction(r, t, s, alpha=np.pi/3):

    # Apply Coordinate Transformation due to stretch
    r = r*np.sqrt(1 + s**3 - (-1 + s**3)*np.cos(2*t))/(np.sqrt(2)*s)

    assert (t >= 0) & (t <= np.pi), "Invalid Polar Angle"
    t = np.pi/2 - np.abs(t-np.pi/2)

    # These were the grids with which the array was generated
    ds = np.linspace(0, 2, 100)
    ts = np.linspace(0, np.pi/2, 100) # only goes to pi/2 with new interpolator, but old one goes to pi
    #alphas = np.linspace(0, np.pi/2, 100)
    #x = (alpha, t, r)
    x = (r, t)
    if r < 2:
        #fraction = interpn((alphas, ts, ds), bandweightfunction, x)[0]  # fraction of points still in band
        #fraction = interpn((ds, ts), bandweightfunction, x)[0]  # fraction of points still in band
        fraction = interp2d(ds, ts, bandweightfunction)(*x)[0]
        bandvolume = 4 * np.pi / 3 * np.cos(alpha)          # dimensionless, i.e. R = 1
        return fraction * bandvolume
    else:
        return 0 

z = 1.0
sv = 0
f = Om0**0.6

def integrand(t, r, rmu, z=z, sv=sv, b=1):
    
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


# Donghui Jeong Liang Dai, Marc Kamionkowski, and Alexander S. Szalay 2014 (Eq 2, 3)
def XiRSDApprox(r, rmu, z=z, sv=sv, b=1, separateterms=False):

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
            Npts = 10000//debug_factor
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
        #b2term = b*b*(xi0*P0(rmu))
        #b1term = b*(2/3*f*xi0*P0(rmu) + 4/3*f*xi2*P2(rmu))
        #b0term = f*f/5*xi0*P0(rmu) + 4/7*f*f*xi2*P2(rmu) + 8/35*f*f*xi4*P4(rmu)
        b2term = xi0*P0(rmu)
        b1term = 2/3*f*xi0*P0(rmu) + 4/3*f*xi2*P2(rmu)
        b0term = f*f/5*xi0*P0(rmu) + 4/7*f*f*xi2*P2(rmu) + 8/35*f*f*xi4*P4(rmu)

        return np.array([b2term, b1term, b0term])
    

        


def XiIntegral(R, s, R1=1e-3, RSD=2, T = [0, np.pi], alpha=None, separateterms=False):
    
    # We add a 2*pi here from the azimuthal integral that would need to be done in the following lines
    if alpha is not None:
        if alphastr == '_alpha':
            print('Using ConeWeightFunction')
            weight = lambda r, t: 2*np.pi*  R*R*R*ConeWeightFunction(r/R, t, s, alpha)
        if alphastr == '_band':
            print('Using BandWeightFunction')
            weight = lambda r, t: 2*np.pi*  R*R*R*BandWeightFunction(r/R, t, s, alpha)
    else:
        weight = lambda r, t: 2*np.pi*TwoEllipsoidCaps(r, t, s, R)

    if RSD == 1:
        assert False, "Invalid, no longer implemented"
    elif RSD == 2:
        if is_root:
            print("Using Dipole + Quadruple O(f^2) Approximation")
        if not separateterms:
            integrand = lambda r, t: weight(r,t)*XiRSDApprox(r,np.cos(t)) * r*r*np.sin(t)
        else:
            integrand = lambda r, t: weight(r,t)*XiRSDApprox(r,np.cos(t), separateterms=separateterms) * r*r*np.sin(t)
    elif RSD == 0:
        integrand = lambda r, t: weight(r,t)*quijote.correlationFunction(r,z) * r*r*np.sin(t)
    elif RSD not in [0,1,2]:
        assert False, "invalid RSD mode"

    dlims = lambda t: [R1, (2*np.sqrt(2)*R*s)/np.sqrt(1 + s**3 + np.cos(2*t) - s**3*np.cos(2*t))]

    #result = nquad(integrand, [dlims, T],opts={'limit': 1000}) # 2D Integral

    # 1D Integral, other done w/ simps
    if not separateterms or RSD == 0:
        def new_integrand(t): 
            lims = dlims(t)
            rgrid = np.linspace(lims[0], lims[1], 1000//debug_factor) # Linear Grid
            #rgrid = np.logspace(np.log10(lims[0]), np.log10(lims[1]), 1000) # Logarithmic Grid
            result = simps([integrand(rg, t) for rg in rgrid], rgrid)
            assert not np.isnan(result), "new_integrand result is nan"
            return result
        
        # Functional integral
        #result = quad(new_integrand, *T) 
        #print(f"Xi({R:.2f}) Integral Error is {np.abs(result[1]/result[0]):.3e}")
        #return result[0]

        # Simpsons Rule
        thgrid = np.linspace(0, np.pi, 200//debug_factor)
        result = simps([new_integrand(th) for th in thgrid], thgrid)
        return result
    else:
        def new_integrand(t):
            lims = dlims(t)
            rgrid = np.linspace(lims[0], lims[1], 1000//debug_factor) # Linear Grid
            #rgrid = np.logspace(np.log10(lims[0]), np.log10(lims[1]), 1000) # Logarithmic Grid
            result = simps([integrand(rg, t) for rg in rgrid], rgrid, axis=0)
            assert not np.isnan(result).all(), "new_integrand result is nan"; assert result.size == 3, "wrong size"
            return result

        # Simpsons Rule
        thgrid = np.linspace(0, np.pi, 100//debug_factor)
        result = simps([new_integrand(th) for th in thgrid], thgrid, axis=0)
        return result




    

'''
To run band vs cone (vs sphere), need to 

1) change ConeWeightFunction/BandWeightFunction
2) change volume (1 - cos) or cos or full sphere
3) change alphastr

'''
if __name__ == '__main__':


    S = [1.0, 0.98, 0.99, 1.01, 1.02]
    n = 4
    R = np.logspace(np.log10(2),np.log10(35),100)*10**((5-n)/3) * 2**(1/3)
    RSD = 2
    results = []
    #alpha = np.pi/2; alphastr = '' #sphere
    alpha = np.pi/3; alphastr = '_alpha' #cone
    #alpha = np.pi/3; alphastr = '_band'  #band

    params = [(s, r) for s in S for r in R]

    xis = {}
    for sto, (s, r) in yt.parallel_objects(params, args.cg, dynamic=True, storage=xis):

        j = params.index((s,r))

        print(f"Rank {cpu_id} is working on item {j}, which is r={r:.3f} Mpc/h for s={s:.3f}")
        start = time()

        sto.result = XiIntegral(r, s, RSD=RSD, alpha=alpha, separateterms=True)
        sto.result_id = f"{j}"

        print(f"Rank {cpu_id} finished in {time()-start:.2f} seconds.")


    CDFs = np.empty((len(R),len(S))); pCDFs = np.empty((len(R),len(S)))
    twoCDFs = np.empty((len(R),len(S))); twopCDFs = np.empty((len(R),len(S)))


    def loss_landscape(opt_b):
        bs = np.linspace(1,5,100)
        losses = np.array([l2(b) for b in bs])
        np.savez(f"loss_landscape_{RSD}_{n}{alphastr}", losses=losses, bs=bs)
        plt.semilogy(bs, losses,label='landscape')
        plt.semilogy(opt_b, np.interp(opt_b, bs, losses),'o', label='optimal')
        plt.title(f'Optimal b = {opt_b:.5f}')
        plt.legend()
        plt.savefig(f"b_loss_landscape_{RSD}_{n}{alphastr}.png")

    if is_root:

        opt_b = 0
        for s in range(len(S)):

            try:
                results = np.array([xis[f"{len(R)*s + j}"] for j in range(len(R))])
            except:
                # Sometimes only works with integer indices??
                results = np.array([xis[len(R)*s + j] for j in range(len(R))])

            nbar = 10**n/1e9

            # Define Volume based on geometry
            if alphastr == '':
                volume = 4*np.pi*R**3/3 #sphere 
            elif alphastr == '_alpha':
                volume = 4*np.pi*R**3/3 * (1 - np.cos(alpha)) #cone
            elif alphastr == '_band':
                volume = 4*np.pi*R**3/3 * np.cos(alpha) #band

            def OneNN(b, originalb=1.0):
                if RSD == 2:
                    return 1 - np.exp(-nbar*volume + nbar*nbar/2*(results * np.array([b*b, b,1])[None,:]).sum(-1))
                elif RSD == 0:
                    return 1 - np.exp(-nbar*volume + nbar*nbar/2*results * b**2 /originalb**2)

            # Only fit bias for s = 1; meaningless otherwise
            if s == 0:
                if alphastr == '_alpha':
                    measurements = np.load("/oak/stanford/orgs/kipac/users/pizza/Quijote/final_cone_cdf.npz")
                    rcb, avg = np.logspace(np.log10(2),np.log10(35), 100)*10**((5-n)/3) * 2**(1/3), measurements['cavg'][5-n, int(RSD==2)] 
                elif alphastr == '_band':
                    measurements = np.load("/oak/stanford/orgs/kipac/users/pizza/Quijote/final_cone_cdf.npz")
                    rcb, avg = np.logspace(np.log10(2),np.log10(35), 100)*10**((5-n)/3) * 2**(1/3), measurements['bavg'][5-n, int(RSD==2)]
                elif alphastr == '':
                    measurements = np.load("final_cdf.npz")
                    rcb, avg = np.logspace(np.log10(2),np.log10(35), 100)*10**((5-n)/3)           , measurements[ 'avg'][5-n, int(RSD==2)]

                def l2(b):
                    if type(b) in [float, np.float32, np.float64]:
                        pass
                    else:
                        b = b[0]
                    onenn = OneNN(b); ponenn = np.minimum(onenn, 1-onenn)
                    m_onenn = np.interp(R, rcb, avg); m_ponenn = np.minimum(m_onenn, 1-m_onenn)
                    loss = ((np.log10(ponenn) - np.log10(m_ponenn))**2)[np.logical_and(m_onenn>0.5,m_onenn<0.9)].mean(0)
                    print(b, loss)
                    return loss

                opt = False
                if opt:
                    opt_b = minimize(l2, np.array([3.0]), method="Nelder-Mead", bounds=(1.0,5.0))
                    assert opt_b.success == True, "Didn't converge (or some other error)"
                    opt_b = opt_b.x[0]

                    print(f"Optimal bias for n={n}, RSD={RSD} is {opt_b:.3f}, done at s={s:.2f}")
                else: 
                    if n == 4:
                        opt_b = 2.5
                    elif n == 3:
                        opt_b = 2.19
                print(f"Using bias {opt_b:.3f} for n={n}, RSD={RSD}")
                loss_landscape(opt_b)
                
    
            def TwoNN(b=opt_b):
                if RSD == 2: 
                    return OneNN(b) - np.exp(-nbar*volume + nbar*nbar/2*(results * np.array([b*b, b,1])[None,:]).sum(-1)) * (nbar*volume - nbar*nbar*(results * np.array([b*b, b,1])[None,:]).sum(-1))
                elif RSD == 0:
                    return OneNN(b) - np.exp(-nbar*volume + nbar*nbar/2*results*b*b) * (nbar*volume - nbar*nbar*results*b*b)
            
            #mnV = -nbar*volume
            #nbsqxiover2 = nbar*nbar/2*results
            #if s==0:
            #    np.savez(f'terms_{RSD}_z={z:.2f}_n={n}{alphastr}', mnV=mnV, nbsqxiover2=nbsqxiover2)

            onenn = OneNN(opt_b)
            onennpcdf = np.minimum(onenn, 1-onenn)

            twonn = TwoNN(opt_b)
            twonnpcdf = np.minimum(twonn, 1-twonn)

            CDFs[:, s]= onenn; pCDFs[:, s]= onennpcdf; twoCDFs[:, s] = twonn; twopCDFs[:, s] = twonnpcdf; 

    CDFs = CDFs.T; pCDFs = pCDFs.T; twoCDFs = twoCDFs.T; twopCDFs = twopCDFs.T;


    if is_root:

        fig, ax = plt.subplots(2,2,figsize=(20,16))

        for q, pcdf in enumerate(pCDFs):
            ax[0][0].loglog(R, pcdf, '.-', label=f"{S[q]:.2f}")
            np.savez(f'pcdf_{RSD}_{q}_n={n}_z={z:.2f}{alphastr}',R=R,pcdf=pcdf, cdf=CDFs[q])
        if alphastr == '_alpha':
            ax[0][0].loglog(rcb, measurements['cpavg'][5-n, int(RSD==2)], '--', label=f"M1")
        elif alphastr == '_band':
            ax[0][0].loglog(rcb, measurements['bpavg'][5-n, int(RSD==2)], '--', label=f"M1")
        elif alphastr == '':
            ax[0][0].log(rcb, measurements['avg'][5-n, int(RSD==2)], '--', label=f"M1")
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


        plt.savefig(f'onennpcdf_{RSD}_{n}_z={z:.2f}{alphastr}_optb.png')





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
