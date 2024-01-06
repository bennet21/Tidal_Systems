import sys
import os
import matplotlib.pyplot as plt
import numpy as np
path =  os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, path)


#**************************************************************************
# Parameters needed for the sediment transport calculations
#**************************************************************************

Ws = 5e-4                # Fall velocity of sediment
alpha = 1e-4             # Erosion coefficent
Kv = 1e-2                # Vertical eddy diffusivity (for vertical mixing)

#**************************************************************************
#           Define time domain
#**************************************************************************

T = (12*60+25)*60        # We model only the M2 and M4 tide. Time is in seconds. 
Tadapt = Kv/Ws**2
Test = 3*Tadapt          # Estimated adaptation time scale to get to equilibrium C
Tnt  = np.ceil(Test/T)
Tend = 3*Tnt*T           # Depends on Ws and Kv. Enough time to Equilibrium C. 
deltaT = 300             # Time step of 5 minutes
t  = np.arange(0, Tend+deltaT, deltaT)
Nt = len(t)

#**************************************************************************
# Prescribed sea surface elevations. It is assumed that d/dx zeta =0. M2
# and M4 are prescribed at the seaward boundary. So these are the sea surface heights in the entire
# basin at any moment in time.
#**************************************************************************

ampD1 = 0            # in part 1 and 2 D1=0. Depending on your estuary, you might want to prescribe D1 for part 3. 
ampM2 = 1
ampM4 = 0.2
phaseD1 = 0
phaseM2 = 0
phaseM4 = np.pi/2

Z = ampD1*np.sin(np.pi*t/T + phaseD1)+ ampM2*np.sin(2*np.pi*t/T + phaseM2)+ ampM4*np.sin(4*np.pi*t/T + phaseM4)          # Waterlevel prescribed as sine function. 
dZdt = ampD1*1*np.pi/T*np.cos(np.pi*t/T+ phaseD1)+ ampM2*2*np.pi/T*np.cos(2*np.pi*t/T+ phaseM2)+ ampM4*4*np.pi/T*np.cos(4*np.pi*t/T + phaseM4)  # Flow velocity will behave as a cosine function. 

#**************************************************************************
#       Spatial Domain and Grid
#**************************************************************************

L  = 1e4                     # We model a simple basin with a length of ten km
dx = 400                     # Grid distance
x  = np.arange(0, L+dx, dx)  # x-coordinate. Seaward end is at x=L, landward end at x=0. 
Nx = len(x)                       
                        
#**************************************************************************
#
#   x=0 (=Inlet) ...................... x=L (=landward side of basin)
#
#   So x=positive in landward direction
#
#   U>0 = Flood flow          U<0 = Ebb flow
#
#**************************************************************************

#**************************************************************************
#           Bed level in basin
#**************************************************************************

H = 10-8e-4 * x             # Bottom profile. Linear sloping bottom. 2 m deep near landward boundary, 10 m deep near inlet. 
dHdx = np.ones(Nx) * -8e-4

#%%
#**************************************************************************
# After a call to hydromodel, flow velocity at each position as a function of
# time is known. This solution is short basin limit. 
#**************************************************************************

from HydroModel import HydroModel

U = HydroModel(t,Z,dZdt,H,dHdx,x,dx);




#%%
#**************************************************************************
# Here you have to calculate the sediment concentrations with the Groen
# model. This is a Matlab function which has as input the flow velocity, the relevant
# parameters, and time. For each position in the basin do a call to this
# Groenmodel. You have to finish the Groen model yourself.
# *************************************************************************
from GroenModel import GroenModel

Cgroen = np.zeros((Nx, Nt))

for px in range(Nx):
    Cgroen[px,0:Nt] = GroenModel(U[px,0:Nt],t,deltaT, T, Ws, alpha, Kv)

Qsgroen = U * Cgroen    # Qs is sediment flux

Nsteps = T/deltaT       # Nr of timesteps in one tidal cycle.


# calculate tidally averaged sediment transport (only averaging over last tidal cycle)
final_cycle = int(-Nsteps)
meanQsgroen = np.mean(Qsgroen[:,final_cycle:], 1)

         
# **************************************************
 
 
# **************************************************************************
#  Use calculated flux with Groen's model to calculate sediment
#  concentration with model that includes advective fluxes
 
#  Make Cmodel yourself.
#  It should include a d/dx UC term, which is actually d/dx Qs
#  You can use code like this
dQsdx, dQsdt = np.gradient(Qsgroen, dx, deltaT)


# **************************************************************************
 #%%
from FullModel import FullModel

Cfull = np.zeros((Nx, Nt))

for px in range(Nx):
    Cfull[px,0:Nt] = FullModel(U[px,0:Nt],t,deltaT, T, Ws, alpha, Kv, dQsdx[px,0:Nt])

 
Qsfull = U * Cfull

final_cycle_1 = int(-Nsteps)
meanQsfull = np.mean(Qsfull[:,final_cycle:], 1) 


#%%
# check for students
# Peak velocities U and Q_green, for for example location 6 (so x = 5)
import scipy.signal
peaks_U = scipy.signal.find_peaks(U[5,:])
peak_U_loc = peaks_U[0]
peak_U = U[5,peak_U_loc]

peaks_Q = scipy.signal.find_peaks(Qsgroen[5,:])
peak_Q_loc = peaks_Q[0]
peak_Q = Qsgroen[5, peak_Q_loc]

# Plot U and Q and show that their peaks are indeed corrent
plt.figure()
plt.plot(U[5,:], label = 'U')
plt.scatter(peak_U_loc, peak_U, label = 'Peaks U')
plt.plot(Qsgroen[5,:], label = 'Qs Green')
plt.scatter(peak_Q_loc, peak_Q, label = 'Peaks Qs Green')
plt.legend()
plt.show()

# Time lag between velocity peaks and concentration peaks
timelag = deltaT * np.mean(abs(peak_U_loc - peak_Q_loc[1:]))  

# 
# For interpolations use Python scipy.interpolate.interp1d

    
