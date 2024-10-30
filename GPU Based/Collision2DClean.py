#2D ion collision simulation code in python. currently there is an issue with low speed collisions that I do not believe was present in earlier
#iterations. Currently trying to debug this.

from IonChainTools import * # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import math
#from sklearn.preprocessing import normalize
import numba
import cupy as cp
from numba import cuda

@numba.njit
def ptovPos(pos,Nmid,dcell):  # converts from physical to virtual units in position
    return (pos/dcell + float(Nmid)) # fractional position in virtual units

@numba.njit
def vtopPos(pos,Nmid,dcell): # converts from virtual to physical units in position (natually placing the point at the center of a cell)
    return float((pos-Nmid))*dcell # returns the center of the cell in physical units

@numba.njit
def ACFields(ErAC0,EzAC0,phaseAC,f,t): # returns AC fields at each grid cell based on the amplitude at each cell, starting phase, current time, and frequency
    return ErAC0*np.sin(phaseAC+f*t*2*np.pi),EzAC0*np.sin(phaseAC+f*t*2*np.pi)

@cuda.jit
def makeRF0_kernel(m, q, w, Nr, Nz, Nrmid, dr, RF):
    i, j = cuda.grid(2)
    if i < Nr and j < Nz:
        C = -m * (w**2) / q
        RF[i, j] = -C * (Nrmid - i) * dr

def makeRF0_gpu(m, q, w, Nr, Nz, Nrmid, dr):
    RF = cp.ones((Nr, Nz))
    threads_per_block = (16, 16)
    blocks_per_grid = ((Nr + threads_per_block[0] - 1) // threads_per_block[0],
                       (Nz + threads_per_block[1] - 1) // threads_per_block[1])
    makeRF0_kernel[blocks_per_grid, threads_per_block](m, q, w, Nr, Nz, Nrmid, dr, RF)
    return RF

@numba.njit
def makeDC(m,q,w,Nz,Nr,Nzmid,dz): # dummy function to generate the DC fields assuming that it is a harmonic potential about (0,0) and focuses in z
    # We take in the mass, frequency for that mass, cell numbers, midpoint, and physical width of a cell and output the DC electric fields (constant in r) as a function of longitudinal cell
    C = -m*(w**2)/q ; DC = np.ones((Nr,Nz))
    for kCell in range(Nz):
        DC[:,kCell] = -DC[:,kCell]*C*(Nzmid-kCell)*dz # electric field for DC in harmonic potential approximation 
    return DC

@numba.njit
def makeVField(m,q,wr,wz,Nr,Nz,Nrmid,Nzmid,dr,dz):
    # we assign voltage at each point given our trapping frequencies
    Cr = -0.5*m*(wr**2)/q ; Cz = -0.5*m*(wz**2)/q ; Vf = np.ones((Nr,Nz))
    for jCell in range(Nr):
        for kCell in range(Nz):
            Vf[jCell,kCell] = Cr*((Nrmid-jCell)*dr)**2 + Cz*((Nzmid-kCell)*dz)**2 # makes a harmonic potential in each axis, adds them
    return Vf

def makeVf(Ni,q,m,l,wr,offsetr,offsetz,vbumpr,vbumpz):
    # this makes an initial array for the ions, the first index is the ion, the second index is r,z,vr,vz,q,m,polarizability [treated as zero]
    vf = np.zeros((Ni,7));
    pos = calcPositions(Ni); lscale = lengthScale(wr); scaledPos = pos*lscale;
    for i in range(Ni):
        vf[i,:] = [0.0e-6,-scaledPos[i],0,0,q,m,0.0]
    vf[l,0] += offsetr ; vf[l,1] += offsetz
    vf[l,2] += vbumpr ; vf[l,3] += vbumpz
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
@numba.njit
def minDists(vf,vc):
    rid2 = 1e6 ; rii2 = 1e6 ; vid2 = 1e6 ; vii2 = 1e6
    Ni = len(vf[:,0]) ; Nc = len(vc[:,0])
    for i in range(Ni):
        for j in range(i+1,Ni): # check each pair of ions for distance and speed
            r = vf[i,0]-vf[j,0] ; z=vf[i,1]-vf[j,1] ; vr = vf[i,2]-vf[j,2] ; vz = vf[i,3]-vf[j,3]
            dist2 = r**2 + z**2
            v2 = vr**2 + vz**2
            if dist2<rii2:
                vii2 = v2 ; rii2 = dist2                
        for j in range(Nc): # check each ion-dipole pair for distance and speed
            r = vf[i,0]-vc[j,0] ; z=vf[i,1]-vc[j,1] ; vr = vf[i,2]-vc[j,2] ; vz = vf[i,3]-vc[j,3]
            dist2 = r**2 + z**2
            v2 = vr**2 + vz**2
            if dist2<rid2:
                vid2 = v2 ; rid2 = dist2                
    return np.sqrt(rid2),np.sqrt(rii2),np.sqrt(vid2),np.sqrt(vii2)

@numba.njit
def collisionMode(rii,rid,a,e=.3):
    return (a*rii**2)/(rid**5)>e

@cuda.jit
def updatePossKernel(vf, dr, dz, dt, Nr, Nz, Nrmid, Nzmid):
    i = cuda.grid(1)
    if i < vf.shape[0]:
        vf[i, 0] += vf[i, 2] * dt
        vf[i, 1] += vf[i, 3] * dt
        rCell = (vf[i, 0] / dr + float(Nrmid))
        zCell = (vf[i, 1] / dz + float(Nzmid))
        if rCell > Nr - 2 or rCell < 1 or zCell > Nz - 2 or zCell < 1:
            vf[i, :] = 0.0
            vf[i, 0] = 2.0
            vf[i, 1] = 2.0
            vf[i, 5] = 1e6

def cuda_updatePoss(vf, dr, dz, dt, Nr, Nz, Nrmid, Nzmid):
    threads_per_block = 256
    blocks = (vf.shape[0] + threads_per_block - 1) // threads_per_block
    updatePossKernel[blocks, threads_per_block](vf, dr, dz, dt, Nr, Nz, Nrmid, Nzmid)
    return vf

@cuda.jit
def updateVelsKernel(vf, Erf, Ezf, dt):
    i = cuda.grid(1)
    if i < vf.shape[0]:
        Fr = vf[i, 4] * Erf[i]
        Fz = vf[i, 4] * Ezf[i]
        vf[i, 2] += Fr * dt / vf[i, 5]
        vf[i, 3] += Fz * dt / vf[i, 5]

def cuda_updateVels(vf, Erf, Ezf, dt, Nrmid, Nzmid):
    threads_per_block = 256
    blocks = (vf.shape[0] + threads_per_block - 1) // threads_per_block
    updateVelsKernel[blocks, threads_per_block](vf, Erf, Ezf, dt)
    return vf

@cuda.jit
def solveFieldsKernel(vf, ErDC, EzDC, ErAC, EzAC, Nrmid, Nzmid, dr, dz, Erf2, Ezf2):
    i = cuda.grid(1)
    if i < vf.shape[0]:
        eps0 = 8.854e-12
        C1 = 4 * np.pi * eps0
        
        jCell = int(round((vf[i, 0] / dr + float(Nrmid))))
        kCell = int(round((vf[i, 1] / dz + float(Nzmid))))
        
        Erf2[i] = ErDC[jCell, kCell] + ErAC[jCell, kCell]
        Ezf2[i] = EzDC[jCell, kCell] + EzAC[jCell, kCell]
        
        for j in range(vf.shape[0]):
            if j != i:
                rdist = vf[j, 0] - vf[i, 0]
                zdist = vf[j, 1] - vf[i, 1]
                sqDist = rdist**2 + zdist**2
                projR = rdist / cp.sqrt(sqDist)
                projZ = zdist / cp.sqrt(sqDist)
                
                Erf2[i] += -projR * vf[j, 4] / (C1 * sqDist)
                Ezf2[i] += -projZ * vf[j, 4] / (C1 * sqDist)

def cuda_solveFields(vf, ErDC, EzDC, ErAC, EzAC, Nrmid, Nzmid, Ni, dr, dz):
    d_vf = cuda.to_device(vf)
    d_ErDC = cuda.to_device(ErDC)
    d_EzDC = cuda.to_device(EzDC)
    d_ErAC = cuda.to_device(ErAC)
    d_EzAC = cuda.to_device(EzAC)
    
    d_Erf2 = cuda.device_array(Ni, dtype=np.float64)
    d_Ezf2 = cuda.device_array(Ni, dtype=np.float64)
    
    threads_per_block = 256
    blocks = (Ni + threads_per_block - 1) // threads_per_block
    
    solveFieldsKernel[blocks, threads_per_block](d_vf, d_ErDC, d_EzDC, d_ErAC, d_EzAC, Nrmid, Nzmid, dr, dz, d_Erf2, d_Ezf2)
    
    return d_Erf2.copy_to_host(), d_Ezf2.copy_to_host()

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
    i = cuda.grid(1)
    Nc, Ni = vc.shape[0], vf.shape[0]
    eps0 = 8.854e-12
    C1 = 4 * np.pi * eps0
    
    if i < Nc:
        # Compute fields on collisional particles
        jCell = int(round((vc[i, 0] / dr + float(Nrmid))))
        kCell = int(round((vc[i, 1] / dz + float(Nzmid))))
        
        Erfc[i, 0] = ErDC[jCell, kCell] + ErAC[jCell, kCell]
        Ezfc[i, 0] = EzDC[jCell, kCell] + EzAC[jCell, kCell]
        
        Erfc[i, 1] = ((ErDC[jCell+1, kCell] + ErAC[jCell+1, kCell]) - (ErDC[jCell-1, kCell] + ErAC[jCell-1, kCell])) / dr
        Ezfc[i, 1] = ((EzDC[jCell, kCell+1] + EzAC[jCell, kCell+1]) - (EzDC[jCell, kCell-1] + EzAC[jCell, kCell-1])) / dz
        
        for j in range(Ni):
            rdist = vf[j, 0] - vc[i, 0]
            zdist = vf[j, 1] - vc[i, 1]
            sqDist = rdist**2 + zdist**2
            projR = rdist / cp.sqrt(sqDist)
            projZ = zdist / cp.sqrt(sqDist)
            
            Erfc[i, 0] += -projR * vf[j, 4] / (C1 * sqDist)
            Ezfc[i, 0] += -projZ * vf[j, 4] / (C1 * sqDist)
            Erfc[i, 1] += 2 * projR * vf[j, 4] / (C1 * sqDist**(3/2))
            Ezfc[i, 1] += 2 * projZ * vf[j, 4] / (C1 * sqDist**(3/2))
        
        # Update collisional particle velocities
        if vc[i, 6] != 0.0:
            pR = -2 * np.pi * eps0 * vc[i, 6] * Erfc[i, 0]
            pZ = -2 * np.pi * eps0 * vc[i, 6] * Ezfc[i, 0]
            pTot = cp.sqrt(pR**2 + pZ**2)
            Fr = abs(pR) * Erfc[i, 1]
            Fz = abs(pZ) * Ezfc[i, 1]
            vc[i, 2] += Fr * dt / vc[i, 5]
            vc[i, 3] += Fz * dt / vc[i, 5]
        
        # Compute fields on ions from collisional particles
        if vc[i, 6] != 0.0:
            for j in range(Ni):
                rdist = vf[j, 0] - vc[i, 0]
                zdist = vf[j, 1] - vc[i, 1]
                dist = cp.sqrt(rdist**2 + zdist**2)
                Rhatr = rdist / dist
                Rhatz = zdist / dist
                Erfi[j] += -abs(pR) * (2 * Rhatr) / (C1 * dist**3)
                Ezfi[j] += -abs(pZ) * (2 * Rhatz) / (C1 * dist**3)

def collisionParticlesFields(vf, vc, Ni, ErDC, EzDC, ErAC, EzAC, dr, dz, dt, Nrmid, Nzmid):
    # Move data to GPU
    d_vf = cuda.to_device(vf)
    d_vc = cuda.to_device(vc)
    d_ErDC = cuda.to_device(ErDC)
    d_EzDC = cuda.to_device(EzDC)
    d_ErAC = cuda.to_device(ErAC)
    d_EzAC = cuda.to_device(EzAC)
    
    # Allocate memory on GPU
    Nc = vc.shape[0]
    d_Erfi = cuda.device_array(Ni, dtype=np.float64)
    d_Ezfi = cuda.device_array(Ni, dtype=np.float64)
    d_Erfc = cuda.device_array((Nc, 2), dtype=np.float64)
    d_Ezfc = cuda.device_array((Nc, 2), dtype=np.float64)
    
    # Launch kernel
    threads_per_block = 256
    blocks = (Nc + threads_per_block - 1) // threads_per_block
    collision_particles_fields_kernel[blocks, threads_per_block](
        d_vf, d_vc, d_ErDC, d_EzDC, d_ErAC, d_EzAC, dr, dz, dt, Nrmid, Nzmid, d_Erfi, d_Ezfi, d_Erfc, d_Ezfc
    )
    
    # Copy results back to host
    vc = d_vc.copy_to_host()
    Erfi = d_Erfi.copy_to_host()
    Ezfi = d_Ezfi.copy_to_host()
    Erfc = d_Erfc.copy_to_host()
    Ezfc = d_Ezfc.copy_to_host()
    
    return vc, Erfi, Ezfi, Erfc, Ezfc


@cuda.jit
def mc_collision_kernel(vf, vc, RF, DC, rs, zs, vrs, vzs, rcolls, zcolls, vrcolls, vzcolls, 
                        Nt, Nr, Nz, dr, dz, dtSmall, dtLarge, dtCollision, eii, eid, reorder):
    i = cuda.grid(1)
    if i >= Nt:
        return

    Ni = vf.shape[0]
    Nc = vc.shape[0]
    Nrmid = (Nr - 1) / 2
    Nzmid = (Nz - 1) / 2
    dtNow = dtSmall

    # Temporary arrays for field calculations
    Erfi = cuda.local.array(shape=(Ni,), dtype=numba.float32)
    Ezfi = cuda.local.array(shape=(Ni,), dtype=numba.float32)

    # Main simulation loop
    for t in range(Nt):
        # Collision detection and dtNow calculation
        rid, rii, vid, vii = minDists(vf, vc)
        collision = collisionMode(rii, rid, vc[0, 6], 0.1)
        
        if collision:
            dtNow = rid * eid / (5 * vid)
        else:
            dtNow = dtSmall
        
        dtNow = max(dtNow, dtCollision)
        
        # Solve fields
        cuda_solveFields(vf, DC, RF, Erfi, Ezfi, Nrmid, Nzmid, Ni, dr, dz)
        
        if vc[0, 5] < 1e6:
            # Collision particles fields
            collisionParticlesFields(vf, vc, Ni, DC, RF, Erfi, Ezfi, dr, dz, dtNow, Nrmid, Nzmid)
            cuda_updatePoss(vc, dr, dz, dtNow, Nr, Nz, Nrmid, Nzmid)
        else:
            dtNow = dtLarge
        
        # Update velocities and positions
        cuda_updateVels(vf, Erfi, Ezfi, dtNow, Nrmid, Nzmid)
        cuda_updatePoss(vf, dr, dz, dtNow, Nr, Nz, Nrmid, Nzmid)
        
        # Update arrays
        for j in range(Ni):
            rs[j, t] = vf[j, 0]
            zs[j, t] = vf[j, 1]
            vrs[j, t] = vf[j, 2]
            vzs[j, t] = vf[j, 3]
        
        if Nc > 0:
            rcolls[0, t] = vc[0, 0]
            zcolls[0, t] = vc[0, 1]
            vrcolls[0, t] = vc[0, 2]
            vzcolls[0, t] = vc[0, 3]
        
        # Check for NaN and update reorder
        if check_nan(vf, vc):
            cuda.atomic.add(reorder, 0, 2)
            break
        
        if check_reorder(zs, t, Ni):
            cuda.atomic.add(reorder, 0, 1)
            break


@cuda.jit(device=True)
def check_nan(vf, vc):
    # Check for NaN values in vf and vc
    for i in range(vf.shape[0]):
        for j in range(vf.shape[1]):
            if not cuda.math.isfinite(vf[i, j]):
                # Reset vf
                for k in range(vf.shape[0]):
                    for l in range(vf.shape[1]):
                        vf[k, l] = 0.0 if l != 5 else 1e1
                # Reset vc
                for k in range(vc.shape[1]):
                    vc[0, k] = 0.0 if k != 5 else 1e1
                return True
    return False

@cuda.jit(device=True)
def check_reorder(zs, t, Ni):
    # Check for reordering condition
    for j in range(1, Ni):
        if zs[j, t] > zs[j-1, t]:
            return True
    return False