import numpy as np
import scipy
import os, configparser
import agama
import matplotlib.pyplot as plt
import math
import emcee
from scipy.optimize import minimize
from multiprocessing import Pool
from scipy import integrate, stats, special

# a separate module which contains the description of the DF model
#from model_DF import createModel
import model_DF

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
    #fudge_fact = (Mlmc/(10**4/232500))**0.1 * 9.
    Rlmc = Mlmc**0.6 * 8.5
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
        sigma = 150/(1+dr/100) #100.0
        couLog= max(0, np.log(dr/Rlmc))
        X     = vmag / (sigma * 2**.5)
        drag  = -4*np.pi * rho / vmag * (scipy.special.erf(X) - 2/np.pi**.5 * X * np.exp(-X*X)) * Mlmc / vmag**2 * couLog
        return np.hstack((v0, f0, v1, f1 + (v1-v0)*drag))

    Tbegin = -3.0  # initial evol time [Gyr]
    Tfinal =  0.   # current time
    Tstep  = 1./128
    tgrid = np.linspace(Tbegin, Tfinal, round((Tfinal-Tbegin)/Tstep)+1)
    ic = np.hstack((np.zeros(6),  # MW
                    [-0.5, -40.8, -27.2, -64.3, -213.4, 208.5]))  # LMC
    sol = scipy.integrate.solve_ivp(difeq, (Tbegin, Tfinal)[::-1], ic, t_eval=tgrid[::-1], max_step=Tstep,
                                    rtol=1e-12, method='LSODA').y.T[::-1]
    '''
    ax=plt.subplots(2,3, figsize=(12,8))[1].reshape(-1)
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
    if not (rr[0]>100): raise RuntimeError('LMC is not unbound')
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

#print(simLMC(agama.Potential('mwslmc1/mwpot.ini')))
#print(simLMC(agama.Potential('mwslmc1/mwpot.ini'), agama.Potential('potentials_triax/lmc00.pot')))
"""
def appr_orb(potmw, potlmc, lmcparams):
    pot_appr = simLMC(potmw, potlmc, lmcparams)
    
    # orbits in the original simulation for a selection of 10000 particles (keep only the first 100 here)
    times = np.load('mwalmc15/df0traj.npz')['times']  # timestamps from -3 Gyr to present (t=0)
    orb_orig = np.load('mwalmc15/df0traj.npz')['trajs'][:200]  # shape: (Norbits, Ntimes, 6)

    #approximate potential
    orb_appr = np.dstack(agama.orbit(potential=pot_appr, ic=orb_orig[:,0],
                time=times[0], timestart=times[-1], trajsize=len(times))[:,1]).swapaxes(1,2).swapaxes(0,1)

    posvel_appr_t3 = orb_appr[0:200,0,:]
    posvel_appr_t0 = orb_appr[0:200,48,:]

    return posvel_appr_t3
"""
##############################################

#function to be minimized:
#sum of squared differences between the values of circular velocity
def circ_vel(params, potmw_data,lmc_mass):

    pot,_,_ = create_potential(params, lmc_mass)
        
    #units of r is kpc
    r_arr = np.logspace(math.log10(2), math.log10(100), 20)
    xyz_arr = np.column_stack((r_arr, r_arr*0, r_arr*0))

    v_circ_data = (np.sqrt(-r_arr * pot.force(xyz_arr)[:,0]))
    v_circ = (np.sqrt(-r_arr * potmw_data.force(xyz_arr)[:,0]))

    return np.sum((v_circ_data-v_circ)**2)

def create_potential(params, lmc_mass):
    #log_M,Mlmc, log_a, alpha, beta, gamma, axisratioz = params
    if lmc_mass == 15:
        log_M, M_lmc, log_a, alpha, beta, gamma, axisratioz = params
        
        M = 10**log_M
        a = 10**log_a

        #extract info from mwpot.ini
        config = configparser.ConfigParser()
        config.read('mwslmc15/mwpot.ini')
        dictionary = {}
    
        for section in config.sections():
            dictionary[section] = {}
            for option in config.options(section):
                dictionary[section][option] = config.get(section, option)
        #print(params)
        params_halo = dict(type = "spheroid",mass=M, scaleradius=a, alpha=alpha, beta=beta, gamma=gamma, axisratioz=axisratioz)
        
        if a > 100 or gamma >= 2 or beta <= 3 or axisratioz > 1.0:
            raise Exception('invalid potential params')

        potmw = agama.Potential(dictionary['Potential bulge'], dictionary['Potential disk'],params_halo)
        potlmc, lmcparams = makeLMC(M_lmc)
      
    else:
        log_M, M_lmc, log_a, alpha, beta, gamma= params

        
        M = 10**log_M
        a = 10**log_a

        if a > 100:
            raise(Exception("Error"))
        
        if (gamma<2) and (beta > 3):
            potmw = agama.Potential(type='Spheroid', mass=M, scaleradius=a, gamma=gamma, beta=beta, alpha=alpha)
            potlmc, lmcparams = makeLMC(M_lmc)
        else:
            raise(Exception("Error"))
        
    return potmw, potlmc, lmcparams

"""
#create DF using given parameter and potential
def create_df(params, pot):
    beta0, log_a_s ,alpha_s, beta_s, gamma_s = params

    a_s = 10**log_a_s

    if (-0.5 <= beta0 < 1) and (0.2 < alpha_s < 5):
        rho_s = agama.Density(type='Spheroid', mass=1, scaleradius=a_s, gamma=gamma_s, beta=beta_s, alpha=alpha_s)
        df = agama.DistributionFunction(type='quasispherical', density=rho_s, potential=pot, beta0=beta0)
    else:
        raise(Exception("Error"))

    return df
"""

def create_df(params, pot, df_type):
    if df_type == "Sph":
        #df_type = 'QuasiSpherical'
        #beta0, log_a_s ,alpha_s, beta_s, gamma_s = params
        df, rho = model_DF.createModel(df_type, params, pot)
    elif df_type == "DPL":
        #df_type = 'DoublePowerLaw'
        #slopeout, slopein, steep, coefJrOut,coefJzOut, coefJrIn, coefJzIn  = params
        df = model_DF.createModel(df_type, params, pot)
    else:
        raise Exception("Wrong Type")
    
    return df
    
#likelihood of DF given the data
def log_prob(params, posvel, pot, df_type, lmc_mass, pot_params):
    try:
        pot,_,_ = create_potential(pot_params, lmc_mass)
        df = create_df(params, pot, df_type)
        # check if the DF is everywhere nonnegative                                                           
        
        j = np.logspace(-2,8,200)
        if any(df(np.column_stack((j, j*0+1e-10, j*0))) <= 0):
            raise Exception("Bad DF")
        
        actfinder = agama.ActionFinder(pot)
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
    
    #extra term penalizing large deviations in non-sph potential
    if lmc_mass == 15:
        solar_radius=8.0 
        solar_vcirc = 233.2
        solar_vcirc_err=10.0
        loglike = np.sum(np.log(df))
        vcirc = (-pot.force(solar_radius, 0, 0)[0] * solar_radius)**0.5
        loglike -= 0.5 * ((vcirc - solar_vcirc) / solar_vcirc_err)**2
        return loglike
    
    else:
        a = np.sum(np.log(df))
        print(a)
        return a

"""
#find the likelihood of the potential+DF parameters given orbits 
def log_probability(params, posvel):
    
    try:
        #check the param size 
        potmw, potlmc, lmcparams = create_potential(params[0:6])
        df = create_df(params[6:], potmw, df_type)

        # check if the DF is everywhere nonnegative
        j = np.logspace(-2,8,200)
        if any(df(np.column_stack((j, j*0+1e-10, j*0))) <= 0):
            raise Exception("Bad DF")
        
        #posvel = appr_orb(potmw, potlmc, lmcparams)

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
        #posvel = appr_orb(pot)
        a = np.sum(np.log(df))
        print(a)
        return a
"""

def main():

    df_type = input("Sph or DPL?")
    lmc_mass = int(input("lmc mass?"))
    df_file = input("df0, df1, df2? ")
    
    #get initial MW+LMC potential data
    potmw = agama.Potential("mwslmc" + str(lmc_mass) + "/mwpot.ini")
    potlmc = agama.Potential('potentials_triax/lmc00.pot')
    lcmparams = dict(type='spheroid',mass = 645000, scaleradius=10.8395,outercutoffradius=108.395,gamma=1,beta=3)
    #lcmparams = polib.pofile('potentials_triax/lmc00.pot')
    
    #intialise guess parameters for potential
    if lmc_mass == 15:
        #Mmw, Mlmc, a, alpha, beta, gamma, axisratio = pot_params
        params = [6.25,1,1,1,10,1,0.3]
    else:
        #Mmw, Mlmc, a, alpha, beta, gamma = pot_params
         params = [6.6,1,1,1,10,1]
    
    #minimize function
    lik_model = minimize(circ_vel, params, args=(potmw,lmc_mass,), method='Nelder-Mead')

    pot_params = lik_model.x
    
    #loading present day orbits
    orb_orig = np.load("mwslmc"+ str(lmc_mass) + "/" + df_file + 'traj.npz')['trajs'][:200]
    posvel_t0 = orb_orig[0:200,48,:]

    #guess params for the DF
    # beta0, log_a_s ,alpha_s, beta_s, gamma_s
    # slopeOut,slopeIn,steepness,coefJrOut,coefJzOut,coefJrIn,coefJzIn,rot,log(j0)

    if df_type == "Sph":
        df_params_guess = [0,1,1,4,1]
    elif df_type == "DPL":
        df_params_guess = [10, 1, 1, 1, 1, 1, 1, 0, 5]
    print(pot_params)
    like_model = minimize(lambda u,v,w,x,y,z : -log_prob(u,v,w,x,y,z), df_params_guess, args=(posvel_t0,potmw,df_type, lmc_mass,pot_params,), method='Nelder-Mead')

    df_params = like_model.x

    params = np.concatenate((pot_params, df_params))
    print(params)
    if os.path.isfile(df_type + '/params' + str(lmc_mass) + "_" + df_file + '.txt') == True:
        os.remove(df_type + '/params' + str(lmc_mass) + "_" + df_file + '.txt')
        file = open(df_type + '/params' + str(lmc_mass) + "_" + df_file + '.txt', 'a')
        for i in range(len(params)):
            file.write(str(params[i]) + "  ")
    else:
        file = open(df_type + '/params' + str(lmc_mass) + "_" + df_file + '.txt', 'a')
        for i in range(len(params)):
            file.write(str(params[i]) + "  ")

main()

"""
##########################################################

    #MCMC routine below

    posx = posvel_t0[:,0]
    posy = posvel_t0[:,1]

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

    Nwalker, Ndim = 50,11

    p0 = params+1.e-4*np.random.randn(50, 11)
    p0[:,10] = np.maximum(0, p0[:,10])

    M_data = (-potmw.force(xyz_arr)[:,0] * r_arr**2)
    v_circ_data = (np.sqrt(-r_arr * potmw.force(xyz_arr)[:,0]) )

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(Nwalker, Ndim, log_probability, args=(posvel_t0,), pool=pool)
        sampler.run_mcmc(p0, 1000, progress=True)

    samples = sampler.get_chain()

    chain = sampler.get_chain(thin=9,discard=100, flat=True)

    for params in chain:
        pot,_,_ = create_potential(params[0:6])
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


    fig.subplots_adjust(wspace=.45)
    ax[0].set_ylabel(r"$log_{10}$($v_{circ}$)")
    ax[0].set_xlabel(r"$log_{10}$(r)")
    ax[1].set_ylabel(r"$log_{10}$(M)")
    ax[1].set_xlabel(r"$log_{10}$(r)")
    ax[0].legend()
    ax[1].legend()
    #plt.savefig("vanilla_df0.png")

    fig, axes = plt.subplots(len(params)+1, figsize=(10, 7), sharex=True)
    #samples = sampler.get_chain()

    labels = ["M_mw","M_lmc", "a", "alpha", "beta", "gamma", "beta0", "a_s", "alpha_s", "beta_s", "gamma_s"]
    for i in range(len(params)):
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
    #plt.savefig("vanilla_df0_emcee.png")
#main()
"""
