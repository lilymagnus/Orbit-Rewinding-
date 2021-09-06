'''
This file is part of the orbit rewind method, and defines the model
specified in terms of an action-based DF f(J).
'''
import agama
import numpy
import math

def createModel(df_type,params, pot):
    '''
    create a model (potential and DF) specified by the given [scaled] parameters
    input params in order:
    -- QuasiSpherical params --
    beta0      (central value of velocity anisotropy)
    lg(a_s)    (anisotropy radius beyond which beta approaches unity [log10-scaled])
    alpha_s      (steepness of the transition between the two asymptotic regimes)
    beta_s       (outer slope)
    gamma_s      (inner slope of density profile of stars)
            
    -- DoublePowerLaw params --
    slopeOut  = ,
    slopeIn   = ,
    steepness = ,
    coefJrOut = ,
    coefJzOut = ,
    coefJrIn  = ,
    coefJzIn  = ,
    rotFrac   = ,
    j0 = 10**,
    '''

    if df_type == 'Sph':
        beta0, log_a_s ,alpha_s, beta_s, gamma_s = params
        a_s = 10**log_a_s
        
        #checking input parameters produce physical DF
        if (-0.5 <= beta0 < 1) and (0.2 < alpha_s < 5) and (beta_s> 3):
            print("ok")
            rho_s = agama.Density(type='Spheroid', mass=1, scaleradius=a_s, gamma=gamma_s, beta=beta_s, alpha=alpha_s)
            df = agama.DistributionFunction(type='quasispherical', density=rho_s, potential=pot, beta0=beta0)
            return df, rho_s
        else:
            return -numpy.inf, -numpy.inf
        
    elif df_type == 'DPL':
        params = params
        # first create an un-normalized DF
        dfparams      = dict(
            type      =     'DoublePowerLaw',
            slopeOut  =     params[0],
            slopeIn   =     params[1],
            steepness =     params[2],
            coefJrOut =     params[3],
            coefJzOut =  params[4],
            coefJrIn  =     params[5],
            coefJzIn  =  params[6],
            rotFrac   = params[7],
            j0        = 10**params[8],
            jphi0        = 0.,
            norm      = 1.)
        # compute its total mass
        totalMass = agama.DistributionFunction(**dfparams).totalMass()
        # and now normalize the DF to have a unit total mass
        dfparams["norm"] = 1. / totalMass
            
        if (params[0] > 3) and (params[1] < 3) and (params[5] > 0) and (params[6] > 0):
            df = agama.DistributionFunction(**dfparams)
            return df
        else:
            return -numpy.inf
        
    else:
        raise(Exception("DF type not found:" + str(df_type)))
