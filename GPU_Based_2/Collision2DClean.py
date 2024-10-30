#2D ion collision simulation code in python. currently there is an issue with low speed collisions that I do not believe was present in earlier
#iterations. Currently trying to debug this.

from IonChainTools import * # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import math
#from sklearn.preprocessing import normalize
import numba
from numba import cuda

@cuda.jit(device=True)
def ptov_pos_device(pos, Nmid, dcell):
    """Device function to convert from physical to virtual units in position."""
    return (pos / dcell + float(Nmid))

@cuda.jit
def ptov_pos_kernel(pos_array, Nmid, dcell, result):
    """CUDA kernel for converting an array of positions from physical to virtual units."""
    i = cuda.grid(1)
    if i < pos_array.shape[0]:
        result[i] = ptov_pos_device(pos_array[i], Nmid, dcell)

@cuda.jit(device=True)
def ptovPos(pos, Nmid, dcell):
    """
    Convert from physical to virtual units in position using CUDA.
    
    Parameters:
    pos : float or numpy array
        Position(s) in physical units
    Nmid : float
        Middle point of the virtual grid
    dcell : float
        Cell size in physical units
    
    Returns:
    float or numpy array
        Position(s) in virtual units
    """
    if isinstance(pos, (int, float)):
        # For single value, use the device function directly
        return ptov_pos_device(pos, Nmid, dcell)
    else:
        # For arrays, use the CUDA kernel
        pos_array = np.asarray(pos)
        pos_gpu = cuda.to_device(pos_array)
        result_gpu = cuda.device_array(pos_array.shape, dtype=pos_array.dtype)
        
        threads_per_block = 256
        blocks = (pos_array.size + threads_per_block - 1) // threads_per_block
        
        ptov_pos_kernel[blocks, threads_per_block](pos_gpu, Nmid, dcell, result_gpu)
        
        return result_gpu.copy_to_host()

@numba.njit
def vtopPos(pos,Nmid,dcell): # converts from virtual to physical units in position (natually placing the point at the center of a cell)
    return float((pos-Nmid))*dcell # returns the center of the cell in physical units

@numba.njit
def ACFields(ErAC0,EzAC0,phaseAC,f,t): # returns AC fields at each grid cell based on the amplitude at each cell, starting phase, current time, and frequency
    return ErAC0*np.sin(phaseAC+f*t*2*np.pi),EzAC0*np.sin(phaseAC+f*t*2*np.pi)

@cuda.jit
def make_rf0_kernel(m, q, w, Nr, Nz, Nrmid, dr, RF):
    """CUDA kernel for generating RF field."""
    j = cuda.grid(1)
    if j < Nr:
        C = -m * (w**2) / q
        for k in range(Nz):
            RF[j, k] = -C * (Nrmid - j) * dr


def makeRF0(m, q, w, Nr, Nz, Nrmid, dr):
    """Generate RF field using CUDA."""
    RF = cuda.device_array((Nr, Nz), dtype=np.float64)
    
    threads_per_block = 256
    blocks = (Nr + threads_per_block - 1) // threads_per_block
    
    make_rf0_kernel[blocks, threads_per_block](m, q, w, Nr, Nz, Nrmid, dr, RF)
    
    return RF.copy_to_host()

@cuda.jit
def make_dc_kernel(m, q, w, Nr, Nz, Nzmid, dz, DC):
    """CUDA kernel for generating DC field."""
    k = cuda.grid(1)
    if k < Nz:
        C = -m * (w**2) / q
        for j in range(Nr):
            DC[j, k] = -C * (Nzmid - k) * dz


def makeDC(m, q, w, Nz, Nr, Nzmid, dz):
    """Generate DC field using CUDA."""
    DC = cuda.device_array((Nr, Nz), dtype=np.float64)
    
    threads_per_block = 256
    blocks = (Nz + threads_per_block - 1) // threads_per_block
    
    make_dc_kernel[blocks, threads_per_block](m, q, w, Nr, Nz, Nzmid, dz, DC)
    
    return DC.copy_to_host()

@numba.njit
def makeVField(m,q,wr,wz,Nr,Nz,Nrmid,Nzmid,dr,dz):
    # we assign voltage at each point given our trapping frequencies
    Cr = -0.5*m*(wr**2)/q ; Cz = -0.5*m*(wz**2)/q ; Vf = np.ones((Nr,Nz))
    for jCell in range(Nr):
        for kCell in range(Nz):
            Vf[jCell,kCell] = Cr*((Nrmid-jCell)*dr)**2 + Cz*((Nzmid-kCell)*dz)**2 # makes a harmonic potential in each axis, adds them
    return Vf

def makeVf(Ni, q, m, l, wr, offsetr, offsetz, vbumpr, vbumpz):
    """
    Make an initial array for the ions using CUDA-optimized subfunctions.
    
    Parameters:
    Ni : int
        Number of ions
    q : float
        Charge of the ions
    m : float
        Mass of the ions
    l : int
        Index of the ion to offset
    wr : float
        Radial trap frequency
    offsetr, offsetz : float
        Offset in r and z directions for ion l
    vbumpr, vbumpz : float
        Velocity bump in r and z directions for ion l
    
    Returns:
    numpy.ndarray
        Array of ion parameters
    """
    # Initialize the vf array on the CPU
    vf = np.zeros((Ni, 7))
    
    # Use the CUDA-optimized calcPositions function
    pos = calcPositions(Ni)
    
    # Use the CUDA-optimized lengthScale function
    lscale = lengthScale(wr)
    
    # Calculate scaled positions
    scaledPos = pos * lscale
    
    # Fill the vf array
    for i in range(Ni):
        vf[i, :] = [0.0e-6, -scaledPos[i], 0, 0, q, m, 0.0]
    
    # Apply offset and velocity bump to the specified ion
    vf[l, 0] += offsetr
    vf[l, 1] += offsetz
    vf[l, 2] += vbumpr
    vf[l, 3] += vbumpz
    
    return vf

@numba.njit
def particleE(vr,vz,m):
    # takes in velocities and mass and returns energy
    return (1/2)*m*(vr**2 + vz**2)

@numba.njit
def particleP(v,m):
    # takes in velocity and mass and returns momentum
    return m*v

@numba.njit
def totalEandP(vrs,vzs,ms):
    # takes in vectors of velocities and masses and returns total energy and momenta
    En = 0.0; pr = 0.0; pz = 0.0
    for i in range(len(vrs)):
        En+=particleE(vrs[i],vzs[i],ms) ; pr+=particleP(vrs[i],ms) ; pz+=particleP(vzs[i],ms)
    return [En,pr,pz]

@numba.njit
def farApart(vf,vc,dCut):
    # we're only checking ion-dipole distances, checks if anything is closer together than dCut
    dCut2 = dCut**2
    Ni = len(vf[:,0]); Nc = len(vc[:,0])
    for i in range(Ni):
        for j in range(Nc):
            d2 = ((vf[i,0]-vc[j,0]))**2 + ((vf[i,1]-vc[j,1]))**2
            if d2 < dCut2:
                return False
    
    return True


def plotThing(Nc,Ni,dt,colls,ions,first,last,title1,xlabel1,ylabel1):
    """
    Makes one continuous plot
    Nc,Ni are the numbers of collisional particles and ions to plot
    first,last are the bounds 
    colls,ions are the arrays of what we want to plot for, like position of each ion over time (we assume time is linear) 
    dt is the time step and the step of the thing we want to plot for
    """
    # Now we plot their positions over time
    timesteps = np.linspace(first,last,last)
    for i in range(Nc):
        plt.plot(dt*timesteps,colls[i,first:last])
    for i in range(Ni):
        plt.plot(dt*timesteps,ions[i,first:last])
    plt.xlabel(xlabel1) ; plt.ylabel(ylabel1) ; plt.title(title1)
    plt.show()    


def plotPieceWise(Nc,Ni,colls1,ions1,colls2,ions2,first,last,title1,xlabel1,ylabel1,xlow,xhigh,ylow,yhigh):
    """
    Makes a scatter plot of any two things against each other
    Nc,Ni are the numbers of collisional particles and ions to plot
    first,last give the slice of the attributes we wish to plot against each other
    colls,ions are the vectors of what we want to plot for 1 denotes the x-axis variable, 2 for the y-axis variable
    title1 is a title, xlabel1 and ylabel1 are axes labels, x/y low/high are axes bounds
    """
    for i in range(Nc):
        plt.scatter(colls1[i,first:last],colls2[i,first:last])
    for i in range(Ni):
        plt.scatter(ions1[i,first:last],ions2[i,first:last])
    plt.xlabel(xlabel1) ; plt.ylabel(ylabel1) ; plt.title(title1)
    plt.xlim(xlow,xhigh) ; plt.ylim(ylow,yhigh)
    plt.show()

def subPlotThings(N1,N2,dt,thing1,thing2,first,last,title1,xlabel1,ylabel1,ylabel2):
    """
    Makes two plots that share an x-axis of time
    N1,N2 are the numbers of vectors of things to plot in each array thing1 and 2 (number of particles)
    first,last are the bounds 
    thing1,thing2 are the vectors of what we want to plot for 
    dt is the time step assuming linear time
    first,last give the slice of thing1 and 2
    title1,x/ylabel1/2 give a title and axes labels
    """
    for i in range(N1):
        plt.plot(dt*range(first,last),thing1[i,first:last])
    for i in range(N2):
        plt.plot(dt*range(first,last),thing2[i,first:last])
    plt.xlabel(xlabel1) ; plt.ylabel(ylabel2) ; plt.title(title1)
    plt.show()

# Define Primary Functions

@cuda.jit
def min_dists_kernel(vf, vc, results):
    """CUDA kernel for calculating minimum distances and velocities."""
    i = cuda.grid(1)
    Ni, Nc = vf.shape[0], vc.shape[0]
    
    if i < Ni:
        rid2, rii2, vid2, vii2 = 1e6, 1e6, 1e6, 1e6
        
        # Check ion-ion distances
        for j in range(i+1, Ni):
            r = vf[i, 0] - vf[j, 0]
            z = vf[i, 1] - vf[j, 1]
            vr = vf[i, 2] - vf[j, 2]
            vz = vf[i, 3] - vf[j, 3]
            dist2 = r**2 + z**2
            v2 = vr**2 + vz**2
            if dist2 < rii2:
                rii2 = dist2
                vii2 = v2
        
        # Check ion-dipole distances
        for j in range(Nc):
            r = vf[i, 0] - vc[j, 0]
            z = vf[i, 1] - vc[j, 1]
            vr = vf[i, 2] - vc[j, 2]
            vz = vf[i, 3] - vc[j, 3]
            dist2 = r**2 + z**2
            v2 = vr**2 + vz**2
            if dist2 < rid2:
                rid2 = dist2
                vid2 = v2
        
        # Use atomic operations to update global minimum values
        cuda.atomic.min(results, 0, rid2)
        cuda.atomic.min(results, 1, rii2)
        cuda.atomic.min(results, 2, vid2)
        cuda.atomic.min(results, 3, vii2)

@cuda.jit(device=True)
def minDists(vf, vc):
    """Calculate minimum distances and velocities using CUDA."""
    Ni = vf.shape[0]
    
    # Prepare GPU arrays
    vf_gpu = cuda.to_device(vf)
    vc_gpu = cuda.to_device(vc)
    results_gpu = cuda.to_device(np.array([1e6, 1e6, 1e6, 1e6], dtype=np.float64))
    
    # Launch kernel
    threads_per_block = 256
    blocks = (Ni + threads_per_block - 1) // threads_per_block
    min_dists_kernel[blocks, threads_per_block](vf_gpu, vc_gpu, results_gpu)
    
    # Copy results back to host and calculate square roots
    results = results_gpu.copy_to_host()
    return (math.sqrt(results[0]), math.sqrt(results[1]), 
            math.sqrt(results[2]), math.sqrt(results[3]))

@cuda.jit(device=True)
def collision_mode_device(rii, rid, a, e):
    """Device function for collision mode calculation."""
    return (a * rii**2) / (rid**5) > e

@cuda.jit
def collision_mode_kernel(rii, rid, a, e, result):
    """CUDA kernel for collision mode calculation."""
    if cuda.grid(1) == 0:
        result[0] = collision_mode_device(rii, rid, a, e)

@cuda.jit(device=True)
def collisionMode(rii, rid, a, e=0.3):
    """Calculate collision mode using CUDA."""
    result_gpu = cuda.device_array(1, dtype=np.bool_)
    collision_mode_kernel[1, 1](rii, rid, a, e, result_gpu)
    return result_gpu.copy_to_host()[0]

@cuda.jit(device=True)
def update_pos_device(vf, dr, dz, dt, Nr, Nz, Nrmid, Nzmid):
    """Device function to update position for a single ion."""
    vf[0] += vf[2] * dt
    vf[1] += vf[3] * dt
    rCell = ptov_pos_device(vf[0], Nrmid, dr)
    zCell = ptov_pos_device(vf[1], Nzmid, dz)
    
    if rCell > Nr - 2 or rCell < 1 or zCell > Nz - 2 or zCell < 1:
        vf[:] = 0.0
        vf[0] = 2.0
        vf[1] = 2.0
        vf[5] = 1e6

@cuda.jit
def update_poss_kernel(vf, dr, dz, dt, Nr, Nz, Nrmid, Nzmid):
    """CUDA kernel to update positions for all ions."""
    i = cuda.grid(1)
    if i < vf.shape[0]:
        update_pos_device(vf[i], dr, dz, dt, Nr, Nz, Nrmid, Nzmid)

@cuda.jit(device=True)
def updatePoss(vf, dr, dz, dt, Nr, Nz, Nrmid, Nzmid):
    """
    This moves our vf particles as their velocity suggests for one time step using CUDA.
    """
    threads_per_block = 256
    blocks = (vf.shape[0] + threads_per_block - 1) // threads_per_block
    
    vf_gpu = cuda.to_device(vf)
    update_poss_kernel[blocks, threads_per_block](vf_gpu, dr, dz, dt, Nr, Nz, Nrmid, Nzmid)
    
    return vf_gpu.copy_to_host()

@cuda.jit
def update_vels_kernel(vf, Erf, Ezf, dt):
    """CUDA kernel for updating velocities."""
    i = cuda.grid(1)
    if i < vf.shape[0]:
        Fr = vf[i, 4] * Erf[i]
        Fz = vf[i, 4] * Ezf[i]
        vf[i, 2] += Fr * dt / vf[i, 5]
        vf[i, 3] += Fz * dt / vf[i, 5]

@cuda.jit(device=True)
def updateVels(vf, Erf, Ezf, dt, Nrmid, Nzmid):
    """Update velocities using CUDA."""
    threads_per_block = 256
    blocks = (vf.shape[0] + threads_per_block - 1) // threads_per_block
    
    vf_gpu = cuda.to_device(vf)
    Erf_gpu = cuda.to_device(Erf)
    Ezf_gpu = cuda.to_device(Ezf)
    
    update_vels_kernel[blocks, threads_per_block](vf_gpu, Erf_gpu, Ezf_gpu, dt)
    
    return vf_gpu.copy_to_host()

@cuda.jit(device=True)
def solve_fields_device(i, vf, ErDC, EzDC, ErAC, EzAC, Nrmid, Nzmid, dr, dz, Erf2, Ezf2):
    """Device function to solve fields for a single ion."""
    eps0 = 8.854e-12
    C1 = 4 * np.pi * eps0
    
    jCell = int(round(ptov_pos_device(vf[i, 0], Nrmid, dr)))
    kCell = int(round(ptov_pos_device(vf[i, 1], Nzmid, dz)))
    
    Erf2[i] = ErDC[jCell, kCell] + ErAC[jCell, kCell]
    Ezf2[i] = EzDC[jCell, kCell] + EzAC[jCell, kCell]
    
    for j in range(vf.shape[0]):
        if j != i:
            rdist = vf[j, 0] - vf[i, 0]
            zdist = vf[j, 1] - vf[i, 1]
            sqDist = rdist**2 + zdist**2
            projR = rdist / math.sqrt(sqDist)
            projZ = zdist / math.sqrt(sqDist)
            Erf2[i] -= projR * vf[j, 4] / (C1 * sqDist)
            Ezf2[i] -= projZ * vf[j, 4] / (C1 * sqDist)

@cuda.jit
def solve_fields_kernel(vf, ErDC, EzDC, ErAC, EzAC, Nrmid, Nzmid, dr, dz, Erf2, Ezf2):
    """CUDA kernel to solve fields for all ions."""
    i = cuda.grid(1)
    if i < vf.shape[0]:
        solve_fields_device(i, vf, ErDC, EzDC, ErAC, EzAC, Nrmid, Nzmid, dr, dz, Erf2, Ezf2)

@cuda.jit(device=True)
def solveFields(vf, ErDC, EzDC, ErAC, EzAC, Nrmid, Nzmid, Ni, dr, dz):
    """
    This solves for the electric fields at each ion from each ion (and the trap) using CUDA.
    """
    Erf2 = np.zeros(Ni)
    Ezf2 = np.zeros(Ni)
    
    threads_per_block = 256
    blocks = (Ni + threads_per_block - 1) // threads_per_block
    
    vf_gpu = cuda.to_device(vf)
    ErDC_gpu = cuda.to_device(ErDC)
    EzDC_gpu = cuda.to_device(EzDC)
    ErAC_gpu = cuda.to_device(ErAC)
    EzAC_gpu = cuda.to_device(EzAC)
    Erf2_gpu = cuda.to_device(Erf2)
    Ezf2_gpu = cuda.to_device(Ezf2)
    
    solve_fields_kernel[blocks, threads_per_block](
        vf_gpu, ErDC_gpu, EzDC_gpu, ErAC_gpu, EzAC_gpu, Nrmid, Nzmid, dr, dz, Erf2_gpu, Ezf2_gpu
    )
    
    Erf2 = Erf2_gpu.copy_to_host()
    Ezf2 = Ezf2_gpu.copy_to_host()
    
    return Erf2, Ezf2

def laserInt(m,I0,nul,sigl,A21,nua,fwhma,bins=2000,width=10,vels=10001,velMax=10):
    """
    This function outputs the interaction strength (resulting in an interaction rate and energy per interaction) between a gaussian laser and a lorentzian atomic transition as an array of frequencies and absorption rates.
    This is equivalent to integrating for each bin over the relevant range between the normalized lineshapes of each component multiplied together.
    This calculates absorption rate as ((c^2*fwhma*A21*I0/(16*sqrt(2)*pi^(5/2)*h*sigl*nu0^3))*Integral(exp(-(nu[1,i]-nul)^2/(2*sigl^2))*((nu[1,i] - nua)^2 + (1.0/4.0)*fwhma^2)^-1,{nu,0,Inf})
   
    nul is the central laser frequency in Hz
    sigl is the gaussian linewidth of the laser in Hz
    nua is the central atomic frequency in Hz
    fwhma is the full width at half maximum of the lorentzian linewidth of the atomic transition in Hz
    vels is the number of velocities to consider for the ion-laser interaction
    bins is the number of frequencies to consider in the given range for interaction strength
    width is the number of frequency widths to consider when binning the interaction strength vector
    A21 is the einstein A coefficient of the atomic transition
    I0 is the laser intensity integrated over frequency [W/m^2]    
    nu is a 2D array where [1,:] is the velocities along the laser direction and [2,:] are the absorption rates for those velocities and [3,:] are the energy differences between absorbed and emitted photon
    """
    h = 6.626e-34 # planck constant [J*s]
    c = 2.998e8 # speed of light [m/s]
    nu = np.zeros((3,bins)) ; nuj = np.zeros((3,vels))
    # println("In laserInt: I0 = ",I0) ; println("nul = ",nul) ; println("sigl = ",sigl)
    # println("A21 = ",A21) ; println("nua = ",nua) ; println("fwhma = ",fwhma)    
    if sigl < fwhma/2.0: # set center frequency and scan width to encompass whole possible lineshape (smaller lineshape dominates)
        d=width*sigl ; nu0=nul
    else:
        d=width*fwhma/2.0 ; nu0=nua
    delta = 2.0*d/bins # step size between bins
    for j in range(vels):
        vel = -velMax + (j-1)*2*velMax/(vels-1) # set velocity for jth bin
        nulj = nul*(1-vel/c)/(1+vel/c) # laser frequency for jth velocity
        nuj[0,j] = vel;
        for i in range(bins): # do the integral
            nuInt = nu0 - d + (i-0.5)*delta # sets frequency to the middle of the bin scanning from nu0 - d to nu0 + d
            gl = exp(-(nuInt-nulj)**2/(2*sigl**2)) # laser lineshape minus constants
            ga = ((nuInt - nua)**2 + (1.0/4.0)*fwhma**2)**(-1) # atomic lineshape minus constants
            nu[1,i] = gl*ga*delta # width of the rectangle with height of the center of the bin for lineshape and width of the bin
        nu[1,:] = nu[1,:]*(c**2*fwhma*A21*I0/(16*np.sqrt(2)*np.pi**(5/2)*h*sigl*nu0**3)) # scale the interaction rate by the various constants
        nuj[1,j] = np.sum(nu[1,:]) #set interaction rate to sum of interaction rates
        if nuj[1,j]>1.0*A21:
            nuj[2,j] = (m/2)*( (h*nulj/(m*c))**2 + (h*nua/(m*c))**2 -2*(nuj[0,j])*(h*nua/(m*c)) + 2*(nuj[0,j])*(h*nulj/(m*c)) -2*(h*nua/(m*c))*(h*nulj/(m*c)) )  # if stimulated emission regime, recoils from absorption in laser direction then recoils from emission against laser direction
        else:
            nuj[2,j] = (m/2)*( (h*nulj/(m*c))**2 + (h*nua/(m*c))**2 + 2*(h*nulj/(m*c))*(nuj[0,j]) ) # if spontaneous emission regime, cools by 
    plt.plot(nuj[0,:],nuj[1,:])
    return nuj

def laserCoolSmoothNew(vf,vl,nu,Ni,dt): # this takes the particle population and applies a global laser cooling field coupled to the radial and axial velocities that purely reduces the velocity like a true damping force
    """
    vf is the vector of ion parameters, the first index is the ion, the second index is r,z,vr,vz,q,m,polarizability
    vl is the laser propagation vector (vr,vz) normalized to 1
    Ni is number of ions, dt is time step
    nu is the array where nu[0,:] are the velocities in the laser direction that interact with the atom and nu[1,;] are the absorption rates of         those frequencies and [2,:] are the energy differences between absorbed and emitted photon
    """

    Nnu = len(nu[0,:])
    #vl = normalize([vl])
    #v_rms = 0.5737442651757222 #rms velocity limit of 40Ca+ for the S1/2 to P1/2 transition in m/s
    for i in range(Ni):
        #magnitude = sqrt(vf[i,3]^2+vf[i,4]^2)
        #if
        #println("ion = ",i)
        if vf[i,5]<1.0: #if the particle exists it should have a realistic mass
            vil = vf[i,2]*vl[0][0]+vf[i,3]*vl[0][1] #velocity of ion in laser propagation direction
            rate = 0.0 ; dE = 0.0
            first = np.searchsorted(nu[0,:],vil)
            #println("first: ", first)
            if first > len(nu[0,:]):
#                println("Error, velocity above scope of laserInt") ; println("index found = ",first) ; println(" velocity = ",vil)
                first = len(nu[0,:])-1 # this assumes the cooling rates at the bounds are sufficiently close to zero
            elif first < 1:
#                println("Error, velocity below scope of laserInt") ; println("index found = ",first) ; println(" velocity = ",vil)   
                first = 0 # this assumes the cooling rates at the bounds are sufficiently close to zero
            rate = nu[1,first] ; dE = nu[2,first]
            #photon_emission_direction = normalize([rand(-1:.01:1),rand(-1:.01:1)]) #random photon emission unit vector
            dv = (2*rate*dt*abs(dE)/vf[i,5])**(1/2) #velocity change determined by the change in kinetic energy
            #println("rate: ", rate, " dE: ", dE, " dt: ", dt, " dv: ", dv)
            #println(dv)
            if dE>0: #the absorbed photon was higher in energy then the emitted photon
                vf[i,2] = vf[i,2]+abs(vl[0][0])*dv*math.copysign(1,vf[i,2]) # radial velocity increase from absorption
                vf[i,3] = vf[i,4]+abs(vl[0][1])*dv*math.copysign(1,vf[i,3]) # axial velocity increase from absorption
            if dE<0:#the absorbed photon was lower in energy then the emitted photon
                vf[i,2] = vf[i,3]-abs(vl[0][0])*dv*math.copysign(1,vf[i,2]) # radial velocity reduction from absorption
                vf[i,3] = vf[i,4]-abs(vl[0][1])*dv*math.copysign(1,vf[i,3]) # axial velocity reduction from absorption
    return vf # we return the updated populations              

@cuda.jit
def collision_particles_fields_kernel(vf, vc, ErDC, EzDC, ErAC, EzAC, dr, dz, dt, Nrmid, Nzmid, Erfi, Ezfi, Erfc, Ezfc):
    eps0 = 8.854e-12
    C1 = 4 * math.pi * eps0
    
    i = cuda.grid(1)
    Nc, Ni = vc.shape[0], vf.shape[0]
    
    if i < Nc:
        # Process collisional particle i
        jCell = int(round(ptov_pos_device(vc[i, 0], Nrmid, dr)))
        kCell = int(round(ptov_pos_device(vc[i, 1], Nzmid, dz)))
        
        Erfc[i, 0] = ErDC[jCell, kCell] + ErAC[jCell, kCell]
        Ezfc[i, 0] = EzDC[jCell, kCell] + EzAC[jCell, kCell]
        Erfc[i, 1] = ((ErDC[jCell+1, kCell] + ErAC[jCell+1, kCell]) - (ErDC[jCell-1, kCell] + ErAC[jCell-1, kCell])) / dr
        Ezfc[i, 1] = ((EzDC[jCell, kCell+1] + EzAC[jCell, kCell+1]) - (EzDC[jCell, kCell-1] + EzAC[jCell, kCell-1])) / dz
        
        for j in range(Ni):
            rdist = vf[j, 0] - vc[i, 0]
            zdist = vf[j, 1] - vc[i, 1]
            sqDist = rdist**2 + zdist**2
            projR = rdist / math.sqrt(sqDist)
            projZ = zdist / math.sqrt(sqDist)
            
            Erfc[i, 0] += -projR * vf[j, 4] / (C1 * sqDist)
            Ezfc[i, 0] += -projZ * vf[j, 4] / (C1 * sqDist)
            Erfc[i, 1] += 2 * projR * vf[j, 4] / (C1 * sqDist**(3/2))
            Ezfc[i, 1] += 2 * projZ * vf[j, 4] / (C1 * sqDist**(3/2))
        
        if vc[i, 6] != 0.0:
            pR = -2 * math.pi * eps0 * vc[i, 6] * Erfc[i, 0]
            pZ = -2 * math.pi * eps0 * vc[i, 6] * Ezfc[i, 0]
            Fr = abs(pR) * Erfc[i, 1]
            Fz = abs(pZ) * Ezfc[i, 1]
            vc[i, 2] += Fr * dt / vc[i, 5]
            vc[i, 3] += Fz * dt / vc[i, 5]
        
        # Process effects on ions
        for j in range(Ni):
            if vc[i, 6] != 0.0:
                rdist = vf[j, 0] - vc[i, 0]
                zdist = vf[j, 1] - vc[i, 1]
                sqDist = rdist**2 + zdist**2
                dist = math.sqrt(sqDist)
                Rhatr = rdist / dist
                Rhatz = zdist / dist
                cuda.atomic.add(Erfi, j, -abs(pR) * (2 * Rhatr) / (C1 * dist**3))
                cuda.atomic.add(Ezfi, j, -abs(pZ) * (2 * Rhatz) / (C1 * dist**3))

@cuda.jit(device=True)
def collisionParticlesFields(vf, vc, Ni, ErDC, EzDC, ErAC, EzAC, dr, dz, dt, Nrmid, Nzmid):
    Nc = len(vc[:, 0])
    
    # Prepare GPU arrays
    vf_gpu = cuda.to_device(vf)
    vc_gpu = cuda.to_device(vc)
    ErDC_gpu = cuda.to_device(ErDC)
    EzDC_gpu = cuda.to_device(EzDC)
    ErAC_gpu = cuda.to_device(ErAC)
    EzAC_gpu = cuda.to_device(EzAC)
    
    Erfi_gpu = cuda.to_device(np.zeros(Ni))
    Ezfi_gpu = cuda.to_device(np.zeros(Ni))
    Erfc_gpu = cuda.to_device(np.zeros((Nc, 2)))
    Ezfc_gpu = cuda.to_device(np.zeros((Nc, 2)))
    
    # Launch kernel
    threads_per_block = 256
    blocks = (Nc + threads_per_block - 1) // threads_per_block
    
    collision_particles_fields_kernel[blocks, threads_per_block](
        vf_gpu, vc_gpu, ErDC_gpu, EzDC_gpu, ErAC_gpu, EzAC_gpu,
        dr, dz, dt, Nrmid, Nzmid, Erfi_gpu, Ezfi_gpu, Erfc_gpu, Ezfc_gpu
    )
    
    # Copy results back to host
    vc = vc_gpu.copy_to_host()
    Erfi = Erfi_gpu.copy_to_host()
    Ezfi = Ezfi_gpu.copy_to_host()
    Erfc = Erfc_gpu.copy_to_host()
    Ezfc = Ezfc_gpu.copy_to_host()
    
    return vc, Erfi, Ezfi, Erfc, Ezfc

@cuda.jit
def mc_collision_kernel(vf, vc, Nt, dtSmall, RF, DC, Nr, Nz, dr, dz, dtLarge, dtCollision, eii, eid, rs, zs, rcolls, zcolls, vrs, vzs, vrcolls, vzcolls, reorder):
    """CUDA kernel for mcCollision simulation"""
    Nrmid = (Nr-1)/2
    Nzmid = (Nz-1)/2
    Ni = vf.shape[0]
    Nc = vc.shape[0]
    
    nullFields = cuda.local.array((Nr, Nz), dtype=np.float64)
    for i in range(Nr):
        for j in range(Nz):
            nullFields[i, j] = 0.0
    
    dtNow = dtSmall
    
    for i in range(Nt):
        # Calculate minimum distances
        rid, rii, vid, vii = minDists(vf, vc)
        
        # Determine collision mode
        collision = collisionMode(rii, rid, vc[0, 6], 0.1)
        
        # Set time step
        if collision:
            dtNow = rid * eid / (5 * vid)
        else:
            dtNow = dtSmall
        
        if dtNow < dtCollision:
            dtNow = dtCollision
        
        # Solve fields
        Erfi, Ezfi = solveFields(vf, nullFields, DC, RF, nullFields, Nrmid, Nzmid, Ni, dr, dz)
        
        if vc[0, 5] < 1e6:
            vc, Erfic, Ezfic, Erfc, Ezfc = collisionParticlesFields(vf, vc, Ni, nullFields, DC, RF, nullFields, dr, dz, dtNow, Nrmid, Nzmid)
            vc = updatePoss(vc, dr, dz, dtNow, Nr, Nz, Nrmid, Nzmid)
            for j in range(Ni):
                Erfi[j] += Erfic[j]
                Ezfi[j] += Ezfic[j]
        else:
            dtNow = dtLarge
        
        # Update velocities and positions
        vf = updateVels(vf, Erfi, Ezfi, dtNow, Nrmid, Nzmid)
        vf = updatePoss(vf, dr, dz, dtNow, Nr, Nz, Nrmid, Nzmid)
        
        # Store results
        for j in range(Ni):
            rs[j, i] = vf[j, 0]
            zs[j, i] = vf[j, 1]
            vrs[j, i] = vf[j, 2]
            vzs[j, i] = vf[j, 3]
        
        for j in range(Nc):
            rcolls[j, i] = vc[j, 0]
            zcolls[j, i] = vc[j, 1]
            vrcolls[j, i] = vc[j, 2]
            vzcolls[j, i] = vc[j, 3]
        
        # Check for NaN values
        if math.isnan(vf[0, 0]):
            for j in range(Ni):
                for k in range(7):
                    vf[j, k] = 0.0 if k < 4 else (1e1 if k == 5 else 0.0)
            for j in range(Nc):
                for k in range(7):
                    vc[j, k] = 0.0 if k < 4 else (1e1 if k == 5 else 0.0)
            break
        
        # Check for ion ejection
        if cuda.atomic.add(reorder, 0, 0) > 1e5:
            cuda.atomic.add(reorder, 0, 2)
            break
        
        # Check for ion reordering
        for j in range(1, Ni):
            if zs[j, i] > zs[j-1, i]:
                cuda.atomic.add(reorder, 0, 1)
                break

def mcCollision(vf, rc, zc, vrc, vzc, qc, mc, ac, Nt, dtSmall, RF, DC, Nr, Nz, dr, dz, dtLarge, dtCollision, eii=0.01, eid=0.01):
    """CUDA-optimized mcCollision function"""
    
    # Initialize arrays
    Ni = vf.shape[0]
    Nc = 1
    vc = np.zeros((Nc, 7))
    vc[0, :] = [rc, zc, vrc, vzc, qc, mc, ac]
    
    rs = cuda.device_array((Ni, Nt), dtype=np.float64)
    zs = cuda.device_array((Ni, Nt), dtype=np.float64)
    vrs = cuda.device_array((Ni, Nt), dtype=np.float64)
    vzs = cuda.device_array((Ni, Nt), dtype=np.float64)
    rcolls = cuda.device_array((Nc, Nt), dtype=np.float64)
    zcolls = cuda.device_array((Nc, Nt), dtype=np.float64)
    vrcolls = cuda.device_array((Nc, Nt), dtype=np.float64)
    vzcolls = cuda.device_array((Nc, Nt), dtype=np.float64)
    
    reorder = cuda.device_array(1, dtype=np.int32)
    
    # Prepare GPU arrays
    vf_gpu = cuda.to_device(vf)
    vc_gpu = cuda.to_device(vc)
    RF_gpu = cuda.to_device(RF)
    DC_gpu = cuda.to_device(DC)
    
    # Launch kernel
    threads_per_block = 256
    blocks = 1
    
    mc_collision_kernel[blocks, threads_per_block](
        vf_gpu, vc_gpu, Nt, dtSmall, RF_gpu, DC_gpu, Nr, Nz, dr, dz, dtLarge, dtCollision, eii, eid,
        rs, zs, rcolls, zcolls, vrs, vzs, vrcolls, vzcolls, reorder
    )
    
    # Copy results back to host
    reorder_value = reorder.copy_to_host()[0]
    
    return reorder_value