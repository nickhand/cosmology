from ..evolution import radius_to_mass

cimport numpy as np
import numpy as np

available_bias = ['Tinker', 'PS', 'SMT']

#-------------------------------------------------------------------------------
def bias_of_mass(mf, bias_model):
    """
    Compute the bias as a function of mass, as specified by the mass functions
    mass array.
    """
    if bias_model not in available_bias:
        raise ValueError("%s is an invalid bias model. Must be one %s" \
                            %(bias_model, available_bias))
    bias_args = (mf.delta_halo,) if bias_model == "Tinker" else ()
    bias_func = globals()['bias_%s' %bias_model]
    b = bias_func(mf.sigma, mf.delta_c, *bias_args)
    
    return b
#-------------------------------------------------------------------------------
def average_halo_bias(k, mf, bias_model, bias_power=1, mass_cut=None):
    """
    Compute the average halo bias corresponding to a given wavenumber, by 
    integrating the halo mass function upwards from mass_cut
    
    Notes
    -----
    As given by Eq. 14 in Bhattacharya and Kosowsky 2008. Given by:
    
    ..math: b^q(k) =  \int dM M dn/dM b(M)^q W^2[k R(M)] / \int dM M dn/dM W^2[k R(M)]
    
    Parameters
    ----------
    k : {float, array_like}
        The wavenumbers to compute the halo bias at [units :math: `h / Mpc`]
    mf : hmf.HaloMassFunction
        The ``HaloMassFunction`` instance to use to compute dn/dm
    bias_model : str
        The name of the bias model to use. Must be one of `bias.available_bias`.
    bias_power : int, optional
        The moment of the bias to compute. Default is 1.
    """
    cdef np.ndarray integral, ks, Rs, spline, N
    
    # first compute R/M limits of integration
    Rmin = 0.01 / np.amax(k)
    Rmax = 10. / np.amin(k)
    
    if mass_cut is None:
        Mmin = radius_to_mass(Rmin, mf.cosmo.mean_dens) # in M_sun/h
    else:
        Mmin = mass_cut
    
    Mmax = radius_to_mass(Rmax, mf.cosmo.mean_dens)
    
    mf.M = np.linspace(np.log10(Mmin), np.log10(Mmax), 1000)
    mf.cut_fit = False
    
    # now compute the bias
    if bias_model not in available_bias:
        raise ValueError("%s is an invalid bias model. Must be one %s" \
                            %(bias_model, available_bias))
    bias_args = (mf.delta_halo,) if bias_model == "Tinker" else ()
    bias_func = globals()['bias_%s' %bias_model]
    b = bias_func(mf.sigma, mf.delta_c, *bias_args)
    
    # set up C arrays to pass
    integral = np.ascontiguousarray(np.empty(len(k)), dtype=np.double)
    ks        = np.ascontiguousarray(k, dtype=np.double)
    Rs        = np.ascontiguousarray(mf.R, dtype=np.double)
    spline   = np.ascontiguousarray(mf.dndlnm*b**bias_power, dtype=np.double)
  
    # do the integral
    avg_bias_integral(<double *>ks.data, len(k), <double *>Rs.data, 
                      <double *>spline.data, len(Rs), Rmin, Rmax, <double *>integral.data)
        
    # and do the normalization
    N      = np.ascontiguousarray(np.empty(len(k)), dtype=np.double)
    spline = np.ascontiguousarray(mf.dndlnm, dtype=np.double)

    avg_bias_integral(<double *>ks.data, len(k), <double *>Rs.data, 
                      <double *>spline.data, len(Rs), Rmin, Rmax, <double *>N.data)
    
    # ignoring constants, since they cancel out here
    return integral/N
#end average_halo_bias
    
#-------------------------------------------------------------------------------
def bias_Tinker(sigmas, delta_c, delta_halo):
    """
    Return the halo bias for the Tinker form.
    
    Tinker, J., et al., 2010. ApJ 724, 878-886.
    http://iopscience.iop.org/0004-637X/724/2/878
    """

    y = np.log10(delta_halo)
    
    # get the parameters as a function of halo overdensity
    A = 1. + 0.24*y*np.exp(-(4./y)**4)
    a = 0.44*y - 0.88
    B = 0.183
    b = 1.5
    C = 0.019 + 0.107*y + 0.19*np.exp(-(4./y)**4)
    c = 2.4
    
    nu = delta_c / sigmas
    return 1. - A * (nu**a)/(nu**a + delta_c**a) + B*nu**b + C*nu**c
#end bias_Tinker

#-------------------------------------------------------------------------------

def bias_PS(sigmas, delta_c):
    """
    Return the halo bias for the Press-Schechter form, as derived by Mo & White
    1996.
    
    Mo, H. J., & White, S. D. M. 1996, MNRAS, 282, 347
    http://adsabs.harvard.edu/abs/1996MNRAS.282..347M
    
    Notes
    -----
    The PS mass function fails to reproduce the dark matter halo mass function
    found in simulations. Bias model overpredicts bias in the range 1 < nu < 3
    and underpredicts at lower masses.
    """
    nu = delta_c / sigmas
    return 1 + (nu**2 - 1)/delta_c
#end bias_PS

#-------------------------------------------------------------------------------
def bias_SMT(sigmas, delta_c):
    """
    Return the halo bias for the Sheth-Mo-Tormen form
    
    Sheth, R. K., Mo, H. J., Tormen, G., May 2001. MNRAS 323 (1), 1-12.
    http://doi.wiley.com/10.1046/j.1365-8711.2001.04006.x
    
    Notes
    -----
    Model underpredicts the clustering of high-peak halos and overpredicts
    the asymptotic bias of low-mass objects. Derived using FOF halos.
    """
    a = 0.707
    b = 0.5
    c = 0.6
    nu = delta_c / sigmas
    
    sqrta = np.sqrt(a)
    term1 = sqrta*(a*nu**2) + sqrta*b*(a*nu**2)**(1-c)
    term2 = (a*nu**2)**c / ((a*nu**2)**c + b*(1-c)*(1-0.5*c))
    return 1 + 1./(sqrta*delta_c)*(term1 - term2)
#end bias_SMT

#-------------------------------------------------------------------------------
    
