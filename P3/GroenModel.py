import numpy as np
from scipy.interpolate import interp1d

def GroenModel(U, t, deltaT, T, Ws, alpha, Kv):
    """ 
    function GroenModel

    INPUT:
        U = 
        t = time
        deltaT = timestep
        T = 
        Ws = 
        alpha = 
        Kv = 

    OUTPUT:
        Conc = 
    """
    
    Tend = t[-1]
    deltaTfix = Kv / Ws**2  # Maximum time step allowed
    
    if deltaTfix > deltaT:
        deltaTfix = deltaT
    else:
        deltaT = deltaTfix
        
    # Within the code deltaT can be time dependent. Most often deltaTfix will be used. 
    
    Nt = int(np.ceil(Tend/deltaT) + 1)
    deltaTnew = Tend / Nt

    tt = np.arange(0, Tend + deltaTnew, deltaTnew)
    
    set_interp = interp1d(t, U, kind = 'linear')
    Uf = set_interp(tt)
    
    E = np.zeros(Nt+1)
    D = np.zeros(Nt+1)
    C = np.zeros(Nt+1)
    
    # initialise solution at t=0
    tt[0] = t[0]
    C[0]  = 0.0
    
    for k in range(Nt):
        
        # ************************************************
        # Predictor step

        E[k] = alpha*Uf[k]**2
        D[k] = (Ws**2/Kv) * C[k]
        # New C will be caused by difference between erosion and deposition
        C[k+1] = C[k] + (E[k]-D[k])*deltaT
        
        # End of predictor step
        # *********************************************
        
        # *****************************************************************
        # Corrector step. See assignment. 
        
        E[k+1] = alpha*Uf[k+1]**2
        D[k+1] = (Ws**2/Kv)*C[k+1]
        C[k+1] = C[k] + 0.5*(E[k]+E[k+1]-D[k]-D[k+1])*deltaT
        
        # End of corrector step
        #*********************************************
        
    # Since we have calculated the solution on a new time vector, we have to interpolate the solution to the right time vector. 
    set_interp = interp1d(tt, C, kind = 'linear')
    Conc = set_interp(t)
    
    return Conc