import numpy as np
import scipy, corner
import os, configparser
import agama
import matplotlib.pyplot as plt
import math
import emcee
from scipy.optimize import minimize
from multiprocessing import Pool
from scipy import stats, integrate, special

############ GLOBAL DF & DENSITY TRUE VALUES ######################
# true potential of the Milky Way
pot_true=agama.Potential(type='spheroid', gamma=1.0, beta=3.5, alpha=0.6, scaleradius= 7.0, densitynorm=1700, outercutoffradius=300)
# density profile most closely resembling the MW globular clusters
den0_true=agama.Density (type='spheroid', gamma=0.0, beta=6.0, alpha=0.5, scaleradius= 6.0, mass=1)
# a more extended density profile
den1_true=agama.Density (type='spheroid', gamma=1.5, beta=4.5, alpha=1.0, scaleradius=15.0, mass=1)
# three choices of DF - df0 is a fiducial DF for clusters, df1 and df2 are somewhat arbitrary
df0_true=agama.DistributionFunction(type='quasispherical', beta0=0.5, potential=pot_true, density=den0_true)
df1_true=agama.DistributionFunction(type='quasispherical', beta0=0.5, potential=pot_true, density=den1_true)
df2_true=agama.DistributionFunction(type='quasispherical', beta0=-.2, potential=pot_true, density=den1_true)

def createDF(params):
    # first create an un-normalized DF
    dfparams      = dict(
        type      = 'DoublePowerLaw',
        slopeOut  = params[0],
        slopeIn   = params[1],
        steepness = params[2],
        coefJrOut = params[3],
        coefJzOut = params[4],
        coefJrIn  = params[5],
        coefJzIn  = params[6],
        rotFrac   = params[7],
        j0        = 10**params[8],
        jphi0     = 0.,
        norm      = 1. )
    totalMass = agama.DistributionFunction(**dfparams).totalMass()
    # and now normalize the DF to have a unit total mass
    dfparams["norm"] = 1. / totalMass
    return agama.DistributionFunction(**dfparams)

df15_true = createDF([8.0, 1.2, 0.6, 0.7, 1.1, 1.7, 1.2, -0.4, 3.9])
#ask about df in mlmc15 file

df_list = [df0_true, df1_true, df15_true, df2_true]


####################### REWIND & LMC TRAJ. PROCEDURES  ###################################

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
    Rlmc = Mlmc**0.6 * 8.5
    fudge_fact = (Mlmc/(10**4/232500))**0.1 * 9.
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
        couLog= max(0, numpy.log(dr / (2*Rlmc)))
        sigma = 150/(1+dr/100) #100.0
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
 
def rewind_orb(potmw, potlmc, lmcparams, lmc_mass, df_file):
    pot = simLMC(potmw, potlmc, lmcparams)
    
    # orbits in the original simulation for a selection of 10000 particles (keep only the first 100 here)
    times = np.load('mwslmc' + str(lmc_mass) +'/' + df_file + 'traj.npz')['times']  # timestamps from -3 Gyr to present (t=0)
    orb_orig = np.load('mwslmc' + str(lmc_mass) +'/' + df_file + 'traj.npz')['trajs'][:200]  # shape: (Norbits, Ntimes, 6)
    
    #rewind the orbits to obtain the present posvels
    orb_rewind = np.dstack(agama.orbit(potential=pot, ic=orb_orig[:,0],
                time=times[0], timestart=times[-1], trajsize=len(times))[:,1]).swapaxes(1,2).swapaxes(0,1)
    
    posvel_rewind_t3 = orb_rewind[0:200,0,:]
    posvel_rewind_t0 = orb_rewind[0:200,48,:]

    return posvel_rewind_t3

#################################################

def data(lmc_mass, df_file):
    #obtaining data from simulation
    
    potmw = agama.Potential("mwslmc" + str(lmc_mass) + "/mwpot.ini")
    potlmc = agama.Potential('potentials_triax/lmc00.pot')
    lcmparams = dict(type='spheroid',mass = 645000, scaleradius=10.8395,outercutoffradius=108.395,gamma=1,beta=3)
    #lcmparams = polib.pofile('potentials_triax/lmc00.pot')

    posvel_rewind = rewind_orb(potmw, potlmc, lcmparams, lmc_mass, df_file)
    
    times = np.load('mwslmc' + str(lmc_mass) +'/' + df_file + 'traj.npz')['times']  # timestamps from -3 Gyr to present (t=0)
    orb_orig = np.load('mwslmc' + str(lmc_mass) +'/' + df_file + 'traj.npz')['trajs'][:200]  # shape: (Norbits, Ntimes, 6)
    posvel_mw_t0 = orb_orig[0:200,48,:]
    posvel_mw_t3 = orb_orig[0:200,0,:]
        
    return potmw, posvel_rewind


#find the likelihood of the paramters given the mock data
def log_probability(params, posvel, df_type, lmc_mass, df_file):
    try:
        #check if potential is spherical
        if lmc_mass == 15:
            potmw,potlmc,lmcparams = create_potential(params[0:7],lmc_mass)
            #check model being used
            if df_type == "Sph":
                df, rho_s =  create_df(params[7:], potmw, df_type)
            else:
                df = create_df(params[7:], potmw, df_type)
        else:
            potmw,potlmc,lmcparams = create_potential(params[0:6],lmc_mass)
            #check model being used
            if df_type == "Sph":
                df, rho_s =  create_df(params[6:], potmw, df_type)
            else:
                df = create_df(params[6:], potmw, df_type)
        
        posvel = rewind_orb(potmw, potlmc,lmcparams, lmc_mass, df_file)
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
def minus_log_probability(params, posvel, df_type, lmc_mass, df_file):
    return  -log_probability(params, posvel, df_type, lmc_mass, df_file)

############ MODEL DF AND POTENTIAL METHODS ################

def create_df(params, pot, df_type):
    if df_type == "Sph":
        #beta0, log_a_s ,alpha_s, beta_s, gamma_s = params
        df, rho = model_DF.createModel(df_type, params, pot)
        return df, rho
    elif df_type == "DPL":
        #breakpoint()
        #slopeout, slopein, steep, coefJrOut,coefJzOut, coefJrIn, coefJzIn, rotFrac, log(j0)  = params
        df = model_DF.createModel(df_type, params, pot)
        return df
    else:
        raise Exception("Wrong Type")
   
#assume the potential is spherical--use most general case: Spheroid
def create_potential(params, lmc_mass):
    #log_M, M_lmc, log_a, alpha, beta, gamma, axisratioz = params
    
    #extract info from mwpot.ini
    if lmc_mass == 15:
        log_M, M_lmc, log_a, alpha, beta, gamma, axisratioz = params
        M = 10**log_M
        a = 10**log_a

        if a > 100:
            raise(Exception("Error"))

        config = configparser.ConfigParser()
        config.read('mwslmc15/mwpot.ini')
        dictionary = {}
        for section in config.sections():
            dictionary[section] = {}
            for option in config.options(section):
                dictionary[section][option] = config.get(section, option)

        params_halo = dict(type="spheroid", mass=M, scaleradius=a, alpha=alpha, beta=beta, gamma=gamma, axisratioz=axisratioz)
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
    

def main():
    
    #prompt to choose DF type + files 
    df_type = input("Sph or DPL? ")
    mass_lmc = int(input("lmc mass: 1, 15 or 2? "))
    df_file = input("df0, df1,or df2? (note lmc15 only has df0): ")
    df_list_index = int(input("choose true df index (0,1,2,3) [df0_true, df1_true, df15_true, df2_true]:")) 
    
    #data from simulation
    pot_data, posvel_data = data(mass_lmc, df_file)

    """
    #params for spherical
    if df_type == "Sph":
        params = [6.25,1,1,1,10,1,0.5,1,1,4,1]
    else:
        #params for doublepowerlaw
        params = [6.25,1,1,1,10,1,10, 1, 1, 1, 1, 1, 1, 0, 5]
    """
    params = np.genfromtxt(df_type + '/params' + str(mass_lmc) + '_' + df_file + '.txt')
     
    posx = posvel_data[:,0]
    posy = posvel_data[:,1]

    """"
    split params into potential and DF params by doing the following
    pot_params = params[0:7]
    df_params = params[7:]
    """
    
    #calculate radii of points to find rmin and rmax
    r = []
    for i in range (len(posx)):
        r.append(np.sqrt(posx[i]**2 + posy[i]**2))
        
    rmin = np.min(r)
    rmax = np.max(r)
    
    r_arr = np.logspace(np.log10(rmin), np.log10(rmax))
    xyz_arr = np.column_stack((r_arr, r_arr*0, r_arr*0))
    
    #lik_model = minimize(minus_log_probability, params, args=(posvel_data, df_type, mass_lmc, df_file,), method='Nelder-Mead')
    #params = lik_model.x
    
    ############### MCMC PROCEDURE ###################

    #move walkers a small distance randomly around the solutions
    Nwalker, Ndim = 50,len(params)
    p0 = params+1.e-4*np.random.randn(50, len(params))

    #sampler = emcee.EnsembleSampler(Nwalker, Ndim, log_probability, args=(posvel_data,df_type,mass_lmc, df_file,))
    #sampler.run_mcmc(p0, 100, progress=True)
    
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(Nwalker, Ndim, log_probability, args=(posvel_data,df_type,mass_lmc, df_file,),pool=pool)
        sampler.run_mcmc(p0, 1100, progress=True)

    samples = sampler.get_chain()    
    #chain = sampler.get_chain(thin=10,discard= 100, flat=True)
    chain = sampler.get_chain(flat=True)

    ################### PLOTTING ##########################

    v_circ_arr = []
    M = []

    M_data = (-pot_data.force(xyz_arr)[:,0] * r_arr**2)
    v_circ_data = (np.sqrt(-r_arr * pot_data.force(xyz_arr)[:,0]) )
    #lists to plot tracer densities
    tracer_density, tracer_sigmar, tracer_sigmat, tracer_anisotropy = [], [], [], []
    #2d array with shape (num_models, num_points_in_radius)
    counter = 0
    for params in chain:
        counter += 1
        #check for spherical potential and create pot + df accordingly
        if mass_lmc ==15:
            pot,_,_ = create_potential(params[0:7], mass_lmc)
            v_circ_arr.append( np.sqrt(-r_arr * pot.force(xyz_arr)[:,0]) )
            M.append(-pot.force(xyz_arr)[:,0] * r_arr**2)

            if df_type == "Sph":
                df, rho_s = create_df(params[7:], pot, df_type)
            else:
                df = create_df(params[7:], pot, df_type)
        else:
            pot,_,_ = create_potential(params[0:6], mass_lmc)
            v_circ_arr.append( np.sqrt(-r_arr * pot.force(xyz_arr)[:,0]) )
            M.append(-pot.force(xyz_arr)[:,0] * r_arr**2)
            if df_type == "Sph":
                df, rho_s = create_df(params[6:], pot, df_type)
            else:
                df = create_df(params[6:], pot, df_type)

        #finding tracers 
        if df_type == "Sph":                
            tracer_density.append(rho_s.density(xyz_arr))    
        else:               
            trdens, trvel2 = agama.GalaxyModel(pot, df).moments(xyz_arr, dens=True, vel=False, vel2=True)
            tracer_density.append(trdens)
            tracer_sigmar.append(trvel2[:,0]**0.5)  # radial velocity dispersion (strictly speaking, root-mean-square radial velocity)
            tracer_sigmar.append((0.5 * (trvel2[:,1] + trvel2[:,2]))**0.5)  # tangential velocity dispersion (average between two other components)
            tracer_anisotropy.append(1 - (trvel2[:,1] + trvel2[:,2]) / (2*trvel2[:,0]) )   # anisotropy parameter                                 
            #selection of models only
            if counter == 50:
                break
    #now v_circ_arr is a 2d array of shape (N_chain, len(r_arr))
    v_circ_median = np.median(v_circ_arr, axis=0)
    M_median = np.median(M, axis=0)
    
    
    #### tracer density ####
    if df_type == "Sph":
        #choosing the correct density profile used in the simulation chosen
        if df_list_index == 0:
            tracer_true = den0_true.density(xyz_arr) 
        else:
            tracer_true = den1_true.density(xyz_arr)

        plt.loglog(r_arr, np.median(tracer_density, axis=0), color = 'r', label = r"Fit")
        plt.loglog(r_arr, tracer_true, label=r'True Potential')
        plt.fill_between(r_arr, np.percentile(tracer_density,16, axis=0), np.percentile(tracer_density,84, axis=0),alpha=0.7)
        plt.fill_between(r_arr, np.percentile(tracer_density,2.3, axis=0), np.percentile(tracer_density,97.7, axis=0),alpha=0.3)
        plt.legend()
        plt.ylabel("Tracer density")
        plt.xlabel("r [kpc]")
        plt.savefig("Sph/tracer" + str(mass_lmc) + df_file + "sph.png")
    else:
        print("save time")

        """
        #reduce computational cost 
        r_arr = np.logspace(0.0, 2.0, 20)  # log-spaced points from 1 to 100 kpc
        xyz_arr = np.column_stack((r_arr, r_arr*0, r_arr*0))
        df = df_list[df_list_index]
        trdens, trvel2 = agama.GalaxyModel(pot_true, df).moments(xyz_arr, dens=True, vel=False, vel2=True)
        tracer_dens_true = trdens
        tracer_sigmar_true = trvel2[:,0]**0.5  # radial velocity dispersion (strictly speaking, root-mean-square radial velocity)
        tracer_sigmat_true = (0.5 * (trvel2[:,1] + trvel2[:,2]))**0.5  # tangential velocity dispersion (average between two other components)
        tracer_anis_true = 1 - (trvel2[:,1] + trvel2[:,2]) / (2*trvel2[:,0])# anisotropy parameter

        fig, ax = plt.subplots(1,2)
        #plot of density and the anisotropy coefficient
        ax[0].loglog(r_arr,np.median(tracer_density,axis=0), color = 'r', label = r"Fit")
        ax[0].loglog(r_arr, tracer_dens_true, label=r'True Potential')
        ax[0].fill_between(r_arr, np.percentile(tracer_density,16, axis=0),np.percentile(tracer_density,84, axis=0), alpha=0.7)
        ax[0].fill_between(r_arr, np.percentile(tracer_density,2.3, axis=0),np.percentile(tracer_density,97.7, axis=0), alpha=0.3)
        ax[0].set_ylabel("Tracer density")
        ax[0].set_xlabel("r [kpc]")

        ax[1].set_ylabel("Tracer anisotropy")
        ax[1].set_xlabel("r [kpc]")
        ax[1].loglog(r_arr,np.median(tracer_anisotropy,axis=0) , color = 'r', label = r"Fit")
        ax[1].loglog(r_arr, tracer_anis_true,label=r'True Potential')
        ax[1].fill_between(r_arr,np.percentile(tracer_anisotropy,16,axis=0),np.percentile(tracer_anisotropy,84,axis=0),alpha=0.7)
        ax[1].fill_between(r_arr,np.percentile(tracer_anisotropy,2.3,axis=0),np.percentile(tracer_anisotropy,97.7,axis=0),alpha=0.3)
        fig.subplots_adjust(wspace=.35)
        ax[0].legend()
        ax[1].legend()
        plt.savefig("DPL/tracer" + str(mass_lmc) + df_file + "DPL.png")
        """
       
    ###### plot mass & circ. vel profiles ########
    
    fig, ax = plt.subplots(1,2)
    
    #v_circ plot
    ax[0].loglog(r_arr, v_circ_median, color = "r", label=r"Fit")
    ax[0].loglog(r_arr, v_circ_data, label=r"True Potential")
    ax[0].fill_between(r_arr, np.percentile(v_circ_arr,16, axis=0),np.percentile(v_circ_arr,84, axis=0), alpha=0.7)
    ax[0].fill_between(r_arr, np.percentile(v_circ_arr,2.3, axis=0),np.percentile(v_circ_arr,97.7, axis=0), alpha=0.3)
    
    #mass plot
    ax[1].loglog(r_arr, M_median, color = "r", label=r"Fit")
    ax[1].loglog(r_arr, M_data, label=r"True Potential")
    ax[1].fill_between(r_arr, np.percentile(M,16, axis=0),np.percentile(M,84, axis=0), alpha=0.7)
    ax[1].fill_between(r_arr,np.percentile(M,2.3, axis=0),np.percentile(M,97.7, axis=0), alpha=0.3)
    
    fig.subplots_adjust(wspace=.35)
    ax[0].set_ylabel(r"$log_{10}$($v_{circ}$)")
    ax[0].set_xlabel(r"$log_{10}$(r)")
    ax[1].set_ylabel(r"$log_{10}$(M)")
    ax[1].set_xlabel(r"$log_{10}$(r)")
    ax[0].legend()
    ax[1].legend()
    if df_type == "Sph":
        plt.savefig("Sph/rewind" + str(mass_lmc) + "_" + df_file + "_sph.png")
    else:
        plt.savefig("DPL/rewind" + str(mass_lmc) + "_" + df_file +"_DPL.png")

    ####### plot walkers in parameter space #####   
    fig, axes = plt.subplots(len(params), figsize=(10, 7), sharex=True)

    if df_type == "Sph":
        labels = [r"$M_mw$",r"$M_lmc$", "a", "alpha", "beta", "gamma",
                  r"$\beta_0$", r"$a_s$", r"$\alpha_s$", r"$\beta_s$", r"$\gamma_s$"]
    else:
        labels = [r"$M_mw$",r"$M_lmc$", "a", "alpha", "beta", "gamma", "slopeOut", "slopeIn",
                  "steepness", "coefJrOut", "coefJzOut", "coefJrIn", "coefJzIn", "rotFrac", "j0"]
        
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
    if df_type == "Sph":
        plt.savefig("Sph/rewind" + str(mass_lmc) + df_file + "_emcee_sph.png")
    else:
        plt.savefig("DPL/rewind" + str(mass_lmc) + df_file + "_emcee_DPL.png")


    ##### corner plots ######
    
    label=['Mmw_true','Mlmc_true','beta0_true']
    true_vals = [1,1,0.5]
    params = params[:,(0,1,6)]
    fig = corner.corner(params, labels=label, truths=true_vals )
    if df_type == "Sph":
        plt.savefig("Sph/corner" + str(mass_lmc) + df_file + "sph.png")
    else:
        plt.savefig("DPL/corner" + str(mass_lmc) + df_file + "DPL.png")
    
main()
        
