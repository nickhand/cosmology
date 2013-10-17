"""
 lensing_power.py
 compute lensing auto and cross power spectra
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 06/29/2013
"""
import numpy as np
from scipy import integrate
import pylab
from cosmology import cosmology, nonlinear_power, linear_power, cosmo
from utils import physical_constants as pc
from utils import utilities, mputils
import multiprocessing as mp

def _power_spectrum_task(k, z, linear, i, pspec_kwargs):
    
    PSpec = nonlinear_power(z, **pspec_kwargs)
    PSpec.set_z(z)
    return PSpec.D2_NL(k) * 2.*np.pi**2 / k**3, i
#end _power_spectrum_task

#-------------------------------------------------------------------------------    
def kernel(z, ni, zlim, c):
    """
    The lensing kernel
    """
    if np.isscalar(ni):
        k = c.lens_kernel(z, ni)
        return k
    else:
        Dm = c.Dm(z)
        A = 1.5*cosmo.omega_m_0*c._H0**2*(1.+z)*Dm/(pc.c_light/pc.km)**2
        def integrand(z_s): 
            Dm_s = c.Dm(z_s)
            return ni(z_s)*(Dm_s - Dm)/Dm_s
        I = integrate.quad(integrand, z, zlim[1])
        return A*I[0]
#end kernel

#-------------------------------------------------------------------------------
def P_ell(ell, 
          nz_1, 
          nz_2 = None, 
          zlim = [0, 10.], 
          Nz = 1000, 
          cosmo_params = None, 
          linear = False, 
          pspec_kwargs = {},
          nprocs = 1):
    """
    The lensing power spectrum, assuming flat universe
    
    Parameters
    -----------
    ell : numpy.ndarray
        the array of multipole numbers to evaluate the power spectrum at
    nz_1 : function or str 
        function giving the 1st redshift distribution or float representing 
        a delta function at that redshift.
    nz_2 : function or float, optional
        function giving the 2nd redshift distribution or float representing 
        a delta function at that redshift. Equals nz_1 if None.
    zlim : list, optional
        list giving the nonzero range of the functions nz_1 and nz_2, 
        for efficiency.
    Nz : int, optional
        the number of redshift bins to use
    cosmo_params : dict, str or cosmology instance, optional
        the cosmology to use
    linear : bool, optional
        whether to use the linear power spectrum
    pspec_kwargs : dict, optional
        the keyword arguments to pass to the halofit.nonlinear_power() class
    nprocs : int, optional
        the number of processors to use in the power spectra calculations, 
        default is one

    Returns
    -------
    ret : numpy.ndarray 
        the spectrum corresponding to the input ell values
    """
    if isinstance(cosmo_params, cosmology):
        c = cosmo_params
    elif cosmo_params is None:
        c = cosmology()
    else:
        c = cosmology(cosmo_params)

    ell = np.asarray(ell)
    
    if zlim[0]==0:
        z = np.logspace(-5, np.log10(zlim[1]),Nz)
    else:
        z = np.logspace(np.log10(zlim[0]), np.log10(zlim[1]), Nz)
    zlim = (z[0],z[-1])
    
    # sample so the output is faster
    c.sample('Dm', np.linspace(0, 1100, 1e3))
    
    # compute kernel for each galaxy distribution
    kern_arrays = []
    kern_arrays.append(np.asarray([kernel(zi, nz_1, zlim, c) for zi in z]))
    if nz_2 is None:
        kern_arrays.append(kern_arrays[0].copy())
    else:
        kern_arrays.append(np.asarray([kernel(zi, nz_2, zlim, c) for zi in z]))
    
    # normalize if the distributions if isn't a delta function
    if not np.isscalar(nz_1):
        norm = integrate.quad(nz_1, zlim[0], zlim[1])[0]
        kern_arrays[0] /= norm
        if nz_2 is None: kern_arrays[1] /= norm
    
    if nz_2 is not None and not np.isscalar(nz_2):
        norm = integrate.quad(nz_2, zlim[0], zlim[1])[0]
        kern_arrays[1] /= norm
    
    # create wavenumbers [1/Mpc] at each redshift bin
    DA = c.Dm(z)
    k = np.zeros( (Nz,len(ell)) )
    k += ell[None,:]
    k /= DA[:,None] 
    
    # compute power spectrum at each redshift bin
    pspecs = np.zeros(k.shape)
    
    # don't do multiprocessing if linear
    if linear:
        PSpec = linear_power()
        
        bar = utilities.initializeProgressBar(Nz)
        for i in range(Nz):
            bar.update(i+1)
            p = PSpec.P_k(k[i], z[i])
            pspecs[i] = p * c.H(z[i]) / DA[i]**2
    else:
        
        # setup multiprocessing
        if nprocs <= mp.cpu_count():
            num_workers = nprocs
        else:
            num_workers = mp.cpu_count()
        num_jobs = Nz
     
        #bar = utilities.initializeProgressBar(Nz)
    
        # initialize the multiprocessing master with nprocs
        master = mputils.mp_master(num_workers, num_jobs, progress=True, log=False)
    
        # enqueue
        for i in range(Nz):
            master.enqueue(mputils.task(_power_spectrum_task, k[i], z[i], linear, i, pspec_kwargs, num=i))
        
        # start the processes
        master.run()
    
        # dequeue the results
        for result in master.dequeue():
            p, num = result
            pspecs[num] = p / DA[num]**2  / c.H(z[num]) * (pc.c_light/pc.km)
        
    # do the final integral over redshift
    kk = kern_arrays[0]*kern_arrays[1]
    integrand = pspecs.copy()*kk[:,None]
    ret = [integrate.simps(z*integrand[:,i], x=np.log(z)) for i, l in enumerate(ell)]
    return ret
#end P_ell

#-------------------------------------------------------------------------------
if __name__ == '__main__':
    import pylab
    
    # define the n(z)
    a, b, c, A = 0.53072693, 7.8100089, 0.51700858, 0.68758185
    n_z = lambda z: A * (z**a + z**(a*b)) / (z**b + c)
   
    # compute the power spectrum
    ell = np.arange(5, 1e4, 1)
    zlim = [0., 10.]
    Pl = P_ell(ell, 0.730, 1.542,   
                    zlim = zlim, 
                    Nz = 1000, 
                    cosmo_params='Planck13',
                    pspec_kwargs = {'use_takahashi' : False}, 
                    linear = True)
    
    # plot the power spectrum
    pylab.loglog(ell, Pl)
    pylab.ylabel(r"$C_\ell$", fontsize=16)
    pylab.xlabel(r"$\ell$", fontsize=16)
    
    #filename = '/Users/Nick/research/analysis/ACT/lensing/analytic/data/clkapCMBkapCMB04282013.dat' 
    #l, cl = np.loadtxt(filename, unpack=True, usecols=[0,1])
    
    #pylab.loglog(l,cl)
    pylab.show()