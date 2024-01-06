import numpy as np

def HydroModel(t, Z, dZdt, H, dHdx, x, dx):
    """ 
    function HydroModel 

    INPUT:
        t = time
        Z = water level at seaward boundary
        dZdt = change of water level in time
        H = still water depth as a function of position x
        L = length of basin
        dx = griddistance

    OUTPUT:
        U = velocity (m/s)
    """
   
    # Initialisation
    Nx = len(x)
    Nt = len(t)
    L = x[-1]
     
    testPlane = np.cumsum(abs(dHdx)>0)
     
    # solve dZdt+d/dx Hu=0, assuming dZdx=0. So U * dHdx + (H+Z) dU/dx=-dZ/dt
    # U=0 at seaward boundary.
    
    U = np.zeros((Nx, Nt))
    if testPlane[-1] == 0:
       H0 = H[0]
       U[0:Nx, 0:Nt] = x * -dZdt/(H0+Z) 
    else:
       U[Nx-1, 0:Nt] = 0
       for px in range(Nx-1):
           U[px, 0:Nt] = (x[px] - L) * (-dZdt/(H[px]+Z))
     
    return U