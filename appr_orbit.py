import numpy as np
import scipy
import os
import agama
import matplotlib.pyplot as plt
import math
import emcee
from scipy.optimize import minimize
from multiprocessing import Pool
from scipy import stats, integrate, special

def makeLMC(Mlmc):
    # input: LMC mass in units of 1e11 Msun (so typical values are between 1 and 2);                
    # the relation between mass and radius is justified in fig.3 of Vasiliev,Belokurov&Erkal 2021.  
    # for creating the Agama potential, the mass unit in natural N-body units is 232500 Msun        
    Rlmc = Mlmc**0.6 * 8.5; Rcut = Rlmc*10
    lmcparams=dict(type='spheroid', gamma=1, beta=3, alpha=1,
                   scaleradius=Rlmc, mass=Mlmc*1e11/232500, outercutoffradius=Rcut)

    model_potlmc = agama.Potential(lmcparams)
    return model_potlmc, lmcparams

def simLMC(potmw, potlmc, lmcparams):
    Mlmc = potlmc.totalMass()
    #recall agama units: 1 solar mass = 232500 & 1kpc = 1                                           
    fudge_fact = (Mlmc/(10**4/232500))**0.6 * 8.5
    def difeq(t, vars):
        x0=vars[0:3]  # MW pos
        v0=vars[3:6]  # MW vel
        x1=vars[6:9]  # LMC pos
        v1=vars[9:12] # LMC vel
        dx=x1-x0
        dr=sum(dx**2)**0.5
        f0=potlmc.force(-dx)
        f1=potmw.force(dx)
        vmag  = sum((v1-v0)**2)**0.5
        rho   = potmw.density(dx)
        sigma = 125/(1+dr/125) #100.0
        couLog= max(0, np.log(dr/30.0))
        X     = vmag / (sigma * 2**.5)
        drag  = -4*np.pi * rho / vmag * (scipy.special.erf(X) - 2/np.pi**.5 * X * np.exp(-X*X)) * Mlmc / vmag**2 * couLog
        return np.hstack((v0, f0, v1, f1 + (v1-v0)*drag))
    
    Tbegin = -3.0  # initial evol time [Gyr]
    Tfinal =  0.   # current time
    Tstep  = 1./128
    tgrid = np.linspace(Tbegin, Tfinal, round((Tfinal-Tbegin)/Tstep)+1)
    ic = np.hstack((np.zeros(6),  # MW
                    [-0.5, -40.8, -27.2, -64.3, -213.4, -208.5]))  # LMC
    sol = scipy.integrate.solve_ivp(difeq, (Tbegin, Tfinal)[::-1], ic, t_eval=tgrid[::-1], max_step=Tstep,
                                    rtol=1e-12, method='LSODA').y.T[::-1]
    '''ax=plt.subplots(2,3, figsize=(12,8))[1].reshape(-1)
    for i in range(6):
        ax[i].plot(trajlmc[:,0], trajlmc[:,i+1], 'b')
        ax[i].plot(tgrid, sol[:,i+6], 'c', dashes=[3,1])
        ax[i].plot(trajmw [:,0], trajmw [:,i+1], 'r')
        ax[i].plot(tgrid, sol[:,i], 'm', dashes=[3,1])
    plt.tight_layout()
    plt.show()
    '''
    rr=np.sum((sol[:,6:9]-sol[:,0:3])**2, axis=1)**0.5
    vr=np.sum((sol[:,6:9]-sol[:,0:3]) * (sol[:,9:12]-sol[:,3:6]), axis=1) / rr
    print('LMC initial distance: %g, vr: %g' % (rr[0],vr[0]))
    # check that the MW is not too heavy, or else the LMC orbit is too small and should be discarded
    if not (np.all(vr[:-16]<0) or rr[0]>200): raise RuntimeError('LMC is not unbound')
    mwx = agama.CubicSpline(tgrid, sol[:,0], der=sol[:,3])
    mwy = agama.CubicSpline(tgrid, sol[:,1], der=sol[:,4])
    mwz = agama.CubicSpline(tgrid, sol[:,2], der=sol[:,5])
    # temp files to store the LMC trajectory and MW acceleration:
    # if you run EMCEE in parallel (with Pool), there will multiple instances of this routine
    # writing the files simultaneously, so their names are randomized
    accfile = 'accel%i.txt' % int(os.getpid())
    lmcfile = 'trajlmc%i.txt' % int(os.getpid())
    # LMC trajectory in the MW-centered reference frame
    trajlmc = np.column_stack((tgrid, sol[:,6:12]-sol[:,0:6]))
    np.savetxt(lmcfile, trajlmc, '%g')
    # MW centre acceleration is minus the second derivative of its trajectory in the inertial frame
    np.savetxt(accfile, np.column_stack((tgrid, -mwx(tgrid,2), -mwy(tgrid,2), -mwz(tgrid,2))), '%g')
    potacc = agama.Potential(type='UniformAcceleration', file=accfile)
    potlmc = agama.Potential(center=lmcfile, **lmcparams)
    # cleanup
    os.remove(accfile)
    os.remove(lmcfile)
    # finally create the total time-dependent potential
    return agama.Potential(potmw, potlmc, potacc)

#print(simLMC(agama.Potential('mwslmc1/mwpot.ini'), agama.Potential('potentials_triax/lmc00.pot'))) 
 
def rewind_orb(potmw, potlmc, lmcparams):
    pot = simLMC(potmw, potlmc, lmcparams)
    
    # orbits in the original simulation for a selection of 10000 particles (keep only the first 100 here)
    times = np.load('mwslmc2/df2traj.npz')['times']  # timestamps from -3 Gyr to present (t=0)
    orb_orig = np.load('mwslmc2/df2traj.npz')['trajs'][:200]  # shape: (Norbits, Ntimes, 6)
    
    #rewind the orbits to obtain the present posvels
    orb_rewind = np.dstack(agama.orbit(potential=pot, ic=orb_orig[:,0],
                time=times[0], timestart=times[-1], trajsize=len(times))[:,1]).swapaxes(1,2).swapaxes(0,1)
    
    posvel_rewind_t3 = orb_rewind[0:200,0,:]
    posvel_rewind_t0 = orb_rewind[0:200,48,:]

    return posvel_rewind_t0

#################################################

def data():
    #obtaining data from simulation
    
    potmw = agama.Potential("mwslmc2/mwpot.ini")
    potlmc = agama.Potential('potentials_triax/lmc00.pot')
    lcmparams = dict(type='spheroid',mass = 645000, scaleradius=10.8395,outercutoffradius=108.395,gamma=1,beta=3)
    #lcmparams = polib.pofile('potentials_triax/lmc00.pot')

    posvel_rewind = rewind_orb(potmw, potlmc, lcmparams)
    
    times = np.load('mwslmc2/df2traj.npz')['times']  # timestamps from -3 Gyr to present (t=0)
    orb_orig = np.load('mwslmc2/df2traj.npz')['trajs'][:200]  # shape: (Norbits, Ntimes, 6)
    posvel_mw_t0 = orb_orig[0:200,48,:]
    posvel_mw_t3 = orb_orig[0:200,0,:]
        
    return potmw, posvel_rewind


#find the likelihood of the paramters given the mock data
def log_probability(params, posvel):
    #print(params)
    #print(posvel[0])
    try:
        potmw,potlmc,lmcparams,df = create_potential(params)
        posvel = rewind_orb(potmw, potlmc,lmcparams)
        # check if the DF is everywhere nonnegative
        j = np.logspace(-2,8,200)
        if any(df(np.column_stack((j, j*0+1e-10, j*0))) <= 0):
            raise Exception("Bad DF")
         
        actfinder = agama.ActionFinder(potmw)
        action = actfinder(posvel)
        
        if np.any(np.isnan(action)):
            print("Error action")
            
    except Exception as E:
        print(E)
        return -np.inf

    df = df(action)
    #check that all energies are negative
    if  np.isnan(df).any():
        #print("fail")
        return -np.inf
    
    else:
        a = np.sum(np.log(df))
        print(a)
        return a
    
    
# function to maximize
def minus_log_probability(params,posvel ):
    return  -log_probability(params, posvel)
    
    
#assume the potential is spherical--use most general case: Spheroid
def create_potential(params):
    #print(params)
        
    #log_M, log_a, alpha, beta, gamma, log_a_s ,alpha_s, beta_s, gamma_s, slopeIn, slopeOut, J0 = params
    log_M,M_lmc, log_a, alpha, beta, gamma, beta0, log_a_s ,alpha_s, beta_s, gamma_s = params
    
    M = 10**log_M
    a = 10**log_a
    a_s = 10**log_a_s
    
    if a > 100:
        raise(Exception("Error"))
    
    #if (M > 0) and (a > 0) and (a_s > 0) and (-0.5 <= beta0 < 1) and (gamma<2) and (0.2 < alpha_s < 5) and (beta > 3):
    #if (slopeIn < 3) and (slopeOut > 3):
    if (-0.5 <= beta0 < 1) and (gamma<2) and (0.2 < alpha_s < 5) and (beta > 3):
        potmw = agama.Potential(type='Spheroid', mass=M, scaleradius=a, gamma=gamma, beta=beta, alpha=alpha)
        potlmc, lmcparams = makeLMC(M_lmc)
        rho_s = agama.Density(type='Spheroid', mass=1, scaleradius=a_s, gamma=gamma_s, beta=beta_s, alpha=alpha_s)
        #df = agama.DistributionFunction(type='DoublePowerLaw', density=rho_s, potential=pot, slopeIn=slopeIn, norm=1, J0=J)
        df = agama.DistributionFunction(type='quasispherical', density=rho_s, potential=potmw, beta0=beta0)
    else:
        raise(Exception("Error"))


    
    return potmw, potlmc,lmcparams, df
    

def main():
        
    #data from simulation
    pot_data, posvel_data = data()
    
    #initialize guess parameters
    #M, a, alpha, beta, gamma, a_s,alpha_s, beta_s, gamma_s, slopeIn, slopeOut, J0 = params
    #log_M,M_lmc, log_a, alpha, beta, gamma, beta0, log_a_s ,alpha_s, beta_s, gamma_s
    
    #params = [6.25,1,1,1,10,1,0.5,1,1,4,1]
    params = np.genfromtxt('params.txt')
    posx = posvel_data[:,0]
    posy = posvel_data[:,1]
    
    #calculate radii of points to find rmin and rmax
    r = []
    for i in range (len(posx)):
        r.append(np.sqrt(posx[i]**2 + posy[i]**2))
        
    rmin = np.min(r)
    rmax = np.max(r)
    
    r_arr = np.logspace(np.log10(rmin), np.log10(rmax))
    xyz_arr = np.column_stack((r_arr, r_arr*0, r_arr*0))
    
    v_circ_arr = []
    M = []

    #lik_model = minimize(minus_log_probability, params, args=(posvel_data), method='Nelder-Mead')

    #print(lik_model)
    #print("start")
    Nwalker, Ndim = 50,11
    
    #move walkers a small distance randomly around the solutions
    p0 = params+1.e-4*np.random.randn(50, 11)
    M_data = (-pot_data.force(xyz_arr)[:,0] * r_arr**2)
    v_circ_data = (np.sqrt(-r_arr * pot_data.force(xyz_arr)[:,0]) )
    
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(Nwalker, Ndim, log_probability, args=(posvel_data,), pool=pool)
        sampler.run_mcmc(p0, 300, progress=True)
        
    samples = sampler.get_chain()
        
    chain = sampler.get_chain(thin=6,discard=100, flat=True)
    
    for params in chain:
        pot,_,_,_ = create_potential(params)
        v_circ_arr.append( np.sqrt(-r_arr * pot.force(xyz_arr)[:,0]) )
        M.append(-pot.force(xyz_arr)[:,0] * r_arr**2)

            
    #now v_circ_arr is a 2d array of shape (N_chain, len(r_arr))
    v_circ_mean = np.mean(v_circ_arr, axis=0)
    v_circ_std  = np.std (v_circ_arr, axis=0)
    
    M_mean = np.mean(M, axis=0)
    M_std = np.std(M, axis=0)
    
    fig, ax = plt.subplots(1,2)
    
    #v_circ plot
    ax[0].loglog(r_arr, v_circ_mean, color = "r", label=r"Fit")
    ax[0].loglog(r_arr, v_circ_data, label=r"True Potential")
    ax[0].fill_between(r_arr, v_circ_mean - v_circ_std, v_circ_mean + v_circ_std, alpha=0.7)
    ax[0].fill_between(r_arr, v_circ_mean - 2*v_circ_std, v_circ_mean + 2*v_circ_std, alpha=0.3)
    
    #mass plot
    ax[1].loglog(r_arr, M_mean,color = "r", label=r"Fit")
    ax[1].loglog(r_arr, M_data, label=r"True Potential")
    ax[1].fill_between(r_arr, M_mean - M_std, M_mean + M_std, alpha=0.7)
    ax[1].fill_between(r_arr, M_mean - 2*M_std, M_mean + 2*M_std, alpha=0.3)
    
    
    fig.subplots_adjust(wspace=.35)
    ax[0].set_ylabel(r"$log_{10}$($v_{circ}$)")
    ax[0].set_xlabel(r"$log_{10}$(r)")
    ax[1].set_ylabel(r"$log_{10}$(M)")
    ax[1].set_xlabel(r"$log_{10}$(r)")
    ax[0].legend()
    ax[1].legend()
    plt.savefig("rewind2_df2_test.png")
    
    fig, axes = plt.subplots(len(params), figsize=(10, 7), sharex=True)
    #samples = sampler.get_chain()
    
    labels = ["M_mw","M_lmc", "a", "alpha", "beta", "gamma", "beta0", "a_s", "alpha_s", "beta_s", "gamma_s"]
    for i in range(len(params)-1):
        ax = axes[i]
        ax.plot(samples[:,:, i], "k", alpha=0.5)
        #ax.set_xlim(0, 100)
        ax.set_ylabel(labels[i])
            
            
    axes[-1].plot(sampler.lnprobability.T, color='k', alpha=0.5)
    axes[-1].set_ylabel('log(L)')
    maxloglike = np.max(sampler.lnprobability)
    axes[-1].set_ylim(maxloglike-3*(len(params)), maxloglike)
    fig.tight_layout(h_pad=0.)
    axes[-1].set_xlabel("step number")
    plt.savefig("rewind1df2_emcee_test.png")
        
main()
        
