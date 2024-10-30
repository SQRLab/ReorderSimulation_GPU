'''Ion String Tools
Provides functions for:
    * Calculating ion positions
    * Calculating vibrational modes
    * Calculating crosstalk intensities
'''

import numpy as np
import math
from scipy.optimize import fsolve, leastsq
from math import exp, pi as π
import scipy.constants as con
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML
from math import sin, pi as π
from numba import cuda

@cuda.jit(device=True)
def ion_position_potential_device(x, m, N):
    """Device function for calculating the potential energy for a single ion."""
    potential = x[m]
    for n in range(m):
        potential -= 1 / (x[m] - x[n])**2
    for n in range(m + 1, N):
        potential += 1 / (x[m] - x[n])**2
    return potential

@cuda.jit
def ion_position_potential_kernel(x, result):
    """CUDA kernel for calculating the potential energy of the ion string."""
    m = cuda.grid(1)
    if m < x.shape[0]:
        result[m] = ion_position_potential_device(x, m, x.shape[0])

def ion_position_potential(x):
    """Calculate the potential energy of the ion string using CUDA."""
    x_gpu = cuda.to_device(np.array(x))
    result_gpu = cuda.device_array(len(x))
    
    threads_per_block = 256
    blocks = (len(x) + threads_per_block - 1) // threads_per_block
    
    ion_position_potential_kernel[blocks, threads_per_block](x_gpu, result_gpu)
    
    return result_gpu.copy_to_host()

def calcPositions(N):
    """Calculate the equilibrium ion positions (remains CPU-based due to fsolve)."""
    from scipy.optimize import fsolve
    estimated_extreme = 0.481 * N**0.765
    return fsolve(ion_position_potential, np.linspace(-estimated_extreme, estimated_extreme, N))

@cuda.jit(device=True)
def length_scale_device(nu, M, Z):
    """CUDA device function for calculating the length scale."""
    elementary_charge = 1.602176634e-19
    epsilon_0 = 8.8541878128e-12
    pi = math.pi
    return ((Z**2 * elementary_charge**2) / (4 * pi * epsilon_0 * M * nu**2))**(1/3)

@cuda.jit
def length_scale_kernel(nu, M, Z, result):
    """CUDA kernel for calculating the length scale."""
    if cuda.grid(1) == 0:
        result[0] = length_scale_device(nu, M, Z)

def lengthScale(nu, M=None, Z=None):
    """Calculate the length scale for the trap using CUDA."""
    if M is None:
        M = 1.660539040e-27 * 39.9626  # atomic_mass * 39.9626
    if Z is None:
        Z = 1
    
    result_gpu = cuda.device_array(1)
    length_scale_kernel[1, 1](nu, M, Z, result_gpu)
    return result_gpu[0]

def calcAxialModes(N, masses=None):
    '''Calculate axial vibrational modes
    
    Params
        N : int
            number of ions
        masses : list
            masses of ions
    
    Returns
        list
        vibrational modes of string, each encoded in a tuple (frequency, mode vector). Frequency is in units of the COM frequency.
    '''
    u = calcPositions(N)
    if masses is None: masses = [1 for _ in range(N)]
    A = [
            [1 + 2*sum([1/abs(u[m]-u[p])**3 if p!=m else 0 for p in range(0, N)]) if n==m
             else -2/abs(u[m]-u[n])**3
             for m in range(0,N)]
        for n in range(0,N)]
    A = np.array(A)
    for i in range(N):
        for j in range(N):
            A[i,j] /= np.sqrt(masses[i])*np.sqrt(masses[j])
    eigvals, eigvecs = np.linalg.eig(A)
    freqs = np.sqrt(eigvals)
    scaledmodes = [(f, v) for f, v in zip(freqs, eigvecs.T)]
    scaledmodes = sorted(scaledmodes, key=lambda mode: mode[0])
    modes = []
    for f, scaledvec in scaledmodes:
        vec = np.array([scaledvec[i]/np.sqrt(masses[i]) for i in range(N)])
        vec = vec / np.sqrt(vec.dot(vec))
        modes.append((f, vec))
    return modes


def calcRadialModes(N, masses=None, νratio=3):
    '''Calculate transverse vibrational modes
    
    Params
        N : int
            number of ions
    
    Returns
        list
        vibrational modes of string, each encoded in a tuple (frequency, mode vector). Frequency is in units of the COM frequency.
    '''
    if masses is None: masses = [1 for _ in range(N)]
    ueq = calcPositions(N)
    A = np.zeros((N, N))
    for i in range(N):
        A[i][i] = -νratio**2 + sum([1/(ueq[i]-ueq[m])**3 for m in range(0, i)]) + sum([1/(ueq[m]-ueq[i])**3 for m in range(i+1, N)])
        for j in range(0, i):
            A[i][j] = -1/(ueq[i]-ueq[j])**3
        for j in range(i+1, N):
            A[i][j] = -1/(ueq[j]-ueq[i])**3
    eigvals, eigvecs = np.linalg.eig(A)
    eigvals *= -1
    invalid_modes = np.where(eigvals<0)
    eigvals = np.delete(eigvals, invalid_modes)
    eigvecs = np.delete(eigvecs, invalid_modes, axis=0)
    freqs = np.sqrt(eigvals)
    scaledmodes = [(f, v) for f, v in zip(freqs, eigvecs.T)]
    scaledmodes = sorted(scaledmodes, key=lambda mode: mode[0])
    modes = []
    for f, scaledvec in scaledmodes:
        vec = np.array([scaledvec[i]/np.sqrt(masses[i]) for i in range(len(eigvals))])
        vec = vec / np.sqrt(vec.dot(vec))
        modes.append((f, vec))
    return modes

def calcCrosstalkIntensities(N, targets, bw):
    '''Calculate crosstalk intensities at ion positions
    
    Params
        N : int
            number of ions
        targets : list
            indices of target ions, which define the centers of the laser beams
        bw : float
            laser beamwidth
    
    Returns
        list
        total laser intensity at ion positions, intensity of 1 corresponding to intensity at center of single beam
    '''
    positions = calcPositions(N)
    beamcenters = [positions[t] for t in targets]
    intensities = calcCrosstalkIntensitiesAtPositions(beamcenters, bw, positions)
    return intensities
    
    
def calcCrosstalkIntensitiesAtPositions(beamcenters, bw, positions):
    '''Calculate crosstalk intensities at given positions
    
    Params
        beamcenters : list
            positions of the centers of each laser beam
        bw : float
            laser beamwidth (w0 beam radius)
        positions : list
            positions at which to evaluate total laser intensity
            
    Returns
        list
        total laser intensity at given positions, intensity of 1 corresponding to intensity at center of single beam
    
    '''
    intensities = sum([np.array([exp(-((p-c)/bw)**2) for p in positions]) for c in beamcenters])**2
    #intensities = sum([np.array([exp(-2*((p-c)/bw)**2) for p in positions]) for c in beamcenters])
    return intensities


## Plotting

def plotPositionsAndCrosstalk(N, targets, bw):
    positions = calcPositions(N)
    beamcenters = [positions[t] for t in targets]
    intensities = calcCrosstalkIntensitiesAtPositions(beamcenters, bw, np.linspace(positions[0]-1, positions[-1]+1, 1000))
    gradient = intensities.reshape(1, -1)
   
    mappable = plt.imshow(gradient, extent=[positions[0]-1, positions[-1]+1, -1, 1], aspect=0.5, cmap='Reds')
    plt.scatter(positions, np.zeros(N))
    plt.colorbar(mappable,orientation='horizontal')#, location='bottom', orientation='horizontal')
    plt.xlabel("Position (units of length scale)")
    plt.tick_params(left = False, labelleft = False)

def animateAxialMode(N, m, masses=None, disp_size=1/3, freq = 2*π/25):
    '''Make an animation of an ion chain moving in a particular axial mode
    
    Params
        N : int
            number of ions in chain
        m : int
            index of vibrational mode (0 is lowest mode)
        masses : list (optional)
            masses of ions, in arbitrary units
        disp_size : float
            size of maximum displacement of ions from equilibrium position, in units of the length scale
        freq : float
            how fast to make the animation, in units of radians/frame
    '''
    fig, ax = plt.subplots()

    ax.set_xlim(( -5, 5))
    ax.set_ylim((-5, 5))
    ax.set_aspect('equal')

    if masses == None:
        masses = [1]*N
    eqpos = calcPositions(N)
    modevec = calcAxialModes(N, masses=masses)[m][1]

    ions = [plt.Circle((0, 0), 0.2*masses[i]**(1/2)) for i in range(N)]

    def init():
        for i, ion in enumerate(ions):
            ion.center = (eqpos[i], 0)
            ax.add_patch(ion)
        ax.scatter(eqpos, np.zeros(N), c="red", s=0.05)
        return ions

    def animate(t):
        for i, ion in enumerate(ions):
            ion.center = (eqpos[i]+sin(freq*t)*modevec[i]*disp_size, 0)
        return ions
    
    anim=animation.FuncAnimation(fig,animate,init_func=init,frames=round(2*π/freq), blit=True)
    plt.close()
    
    return HTML(anim.to_jshtml())


def animateRadialMode(N, m, masses=None, disp_size=1/3, freq = 2*π/25):
    '''Make an animation of an ion chain moving in a particular axial mode
    
    Params
        N : int
            number of ions in chain
        m : int
            index of vibrational mode (0 is lowest mode)
        masses : list (optional)
            masses of ions, in arbitrary units
        disp_size : float
            size of maximum displacement of ions from equilibrium position, in units of the length scale
        freq : float
            how fast to make the animation, in units of radians/frame
    '''
    fig, ax = plt.subplots()

    ax.set_xlim(( -5, 5))
    ax.set_ylim((-5, 5))
    ax.set_aspect('equal')

    if masses == None:
        masses = [1]*N
    eqpos = calcPositions(N)
    modevec = calcRadialModes(N, masses=masses,νratio = 10)[m][1]

    ions = [plt.Circle((0, 0), 0.2*masses[i]**(1/2)) for i in range(N)]

    def init():
        for i, ion in enumerate(ions):
            ion.center = (eqpos[i], 0)
            ax.add_patch(ion)
        ax.scatter(eqpos, np.zeros(N), c="red", s=0.05)
        return ions

    def animate(t):
        for i, ion in enumerate(ions):
            ion.center = (eqpos[i],sin(freq*t)*modevec[i]*disp_size)
        return ions
    
    anim=animation.FuncAnimation(fig,animate,init_func=init,frames=round(2*π/freq), blit=True)
    plt.close()
    
    return HTML(anim.to_jshtml())

def visualizeAxialMode(N, m, ax, masses=None,disp_size=1/3.,ion_size=20,head_size = 0.1):
    
    if masses == None:
        masses = [1]*N
        
    eqpos = calcPositions(N)
    modevec = calcAxialModes(N,masses=masses)[m][1]
    
    for i in range(N):
        ax.plot(eqpos[i],0,'.b',markersize=ion_size*masses[i]**(1/2))
        if abs(modevec[i]) > 1e-8:
            ax.arrow(eqpos[i],0,modevec[i],0,length_includes_head=True,head_width=head_size, head_length=0.1,shape="full", color='k')
    
    ax.set_xlim(min(eqpos*1.5),max(eqpos*1.5))
    
    return ax


def visualizeRadialMode(N, m, ax, masses=None,disp_size=0.1,ion_size=20,head_size=0.25):
    
    if masses == None:
        masses = [1]*N
        
    eqpos = calcPositions(N)
    modevec = calcRadialModes(N,masses=masses)[m][1]
    
    for i in range(N):
        ax.plot(eqpos[i],0,'.b',markersize=ion_size*masses[i]**(1/2))
        if abs(modevec[i]) > 1e-8:
            ax.arrow(eqpos[i],0,0,modevec[i]*disp_size,width=disp_size*0.25,length_includes_head=True,head_width=head_size*(max(eqpos)-min(eqpos))/N, head_length=disp_size*0.1,shape="full", color='k')
    
    ax.set_xlim(min(eqpos*1.5),max(eqpos*1.5))
    ax.set_ylim(-0.1,0.1)
    
    return ax

def axialfreq_from_minimum_spacing(N, M, minspacing):
    '''Calculate the axial frequency necessary to achieve a particular minimum value for inter-ion spacing.

    Params
        N (integer): Number of ions in the string
        M (float): Mass of ion species, in kilograms
        minspacing (float): Minimum spacing between ions, in meters

    Returns
        (float): Axial frequency such that ions will have given minimum spacing
    '''
    ionpositions_scaledunits = calcPositions(N)
    minspacing_scaledunits = np.min([ionpositions_scaledunits[i+1]-ionpositions_scaledunits[i]  for i in range(N-1)])
    desired_lengthscale = minspacing/minspacing_scaledunits
    reference_lengthscale = lengthScale(2*π*1e6, M=M) # Lengthscale
    necessary_axialfreq = (desired_lengthscale/reference_lengthscale)**(-3/2)*2*π*1e6
    return necessary_axialfreq

def trap_freq_ratio_for_stablility(N):
    '''Calculate the minimum ratio between the radial and axial trap frequencies
    necessary for a stable ion string. Value returned will be at most 0.001 more
    than actual value.
    
    Params
        N (int) : Length of string
        
    Returns
        (float) Minimum ratio of (radial trap frequency)/(axial trap frequency) necessary for string stability
    '''
    approx_min_ratio = 0
    for stepsize in (10,1,0.1,0.01,0.001):
        for ratio in np.arange(approx_min_ratio, approx_min_ratio+11*stepsize, stepsize):
            #print(N, ratio)
            if len(calcRadialModes(N, νratio=ratio)) == N:
                #print('over')
                approx_min_ratio = ratio-stepsize
                break
    approx_min_ratio += 0.001
    return approx_min_ratio


def Ωvals_from_gaussian_crosstalk(N, targets, νz, M, beamradius):
    '''Calculate rabi frequency values (proportional to electric field strength) present
    the sites of the ions based on a gaussian beam profile.

    Params
        N (int) : Number of ions the string
        targets (tuple) : Indices of ions on which laser beams are centered
        νz (float) : Axial trap frequency (radian Hz) 
        M (float) : Mass of ion species (kg)
        beamradius (float) : Distance from beam center at which electric field intensity goes down by 1/e

    Returns
        (numpy array) N rabi frequency values, scaled so the center of a single beam corresponds to 1
    '''
    positions = calcPositions(N)*lengthScale(νz, M=M)
    gaussian = lambda d : exp(-(d/beamradius)**2)
    Ωvals = [sum(gaussian(positions[t]-positions[i]) for t in targets if gaussian(positions[t]-positions[i])>0.01) for i in range(N)]
    return np.array(Ωvals)
