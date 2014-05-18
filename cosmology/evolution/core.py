from ..parameters import Cosmology, default_params
from ..utils import constants as c

import numpy as np
import scipy.integrate as integ 

#-------------------------------------------------------------------------------
def hubble_time():
    """
    The Hubble time, 1/H0 [units: Gyr/h]
    """
    return (1./100.) / (c.km / c.second / c.Mpc) / (c.giga * c.year)
#end hubble_time

#-------------------------------------------------------------------------------
def hubble_distance():
    """
    The Hubble distance, c/H0 [units: Mpc/h]
    """
    return (1./100.) * c.c_light / (c.km / c.second) 
#end hubble_distance

#-------------------------------------------------------------------------------
def H(z, params=default_params):
    """
    The value of the Hubble constant at redshift z [units: km/s/Mpc]
    
    Parameters
    ----------
    z : {float, array_like}
        the redshift to compute the function at
    params : {str, dict, Cosmology}
        the cosmological parameters to use. Default is set by 
        the value of ``parameters.default_params``
    """
    if not isinstance(params, Cosmology):
        params = Cosmology(params)
    return params.H0*E(z, params=params)
#end H

#-------------------------------------------------------------------------------
@np.vectorize
def E(z, params=default_params):
    """
    The unitless Hubble expansion rate at redshift z, 
    modified to include the dark energy equation of state w. 

    Parameters
    ----------
    z : {float, array_like}
        the redshift to compute the function at
    params : {str, dict, Cosmology}
        the cosmological parameters to use. Default is set by 
        the value of ``parameters.default_params``
    """
    if not isinstance(params, Cosmology):
        params = Cosmology(params)

    return np.sqrt(params.omegam*(1.+z)**3 + params.omegar*(1.+z)**4 \
                    + params.omegak*(1.+z)**2 \
                    + params.omegal*(1.+ z)**(3.*(1+params.w)))
#end E

#-------------------------------------------------------------------------------
@np.vectorize
def lookback_time(z, params=default_params):
    """
    The lookback time, defined as the difference between the age 
    of the Universe now and the age at z. 
    
    Notes
    -----
    As given in Eq. 30 of Hogg 1999. 
    
    Parameters
    ----------
    z : {float, array_like}
        The redshift to compute the lookback time to
    params : {str, dict, Cosmology}
        the cosmological parameters to use. Default is set by 
        the value of ``parameters.default_params``
    """
    if not isinstance(params, Cosmology):
        params = Cosmology(params)
        
    f = lambda z: 1/(1.+z)/E(z, params=params)
    I = integ.quad(f, 0, z)

    return  (hubble_time() / params.h) * I[0]
#end lookback_time
    
#-------------------------------------------------------------------------------
@np.vectorize
def age(z, params=default_params):
    """
    The age of the universe at redshift z [units: Gyr]
    
    Parameters
    ----------
    z : {float, array_like}
        the redshift to compute the age of the Universe at
    params : {str, dict, Cosmology}
        the cosmological parameters to use. Default is set by 
        the value of ``parameters.default_params``
    """
    if not isinstance(params, Cosmology):
        params = Cosmology(params)
        
    f = lambda z: 1/(1.+z)/E(z, params=params)
    I = integ.quad(f, z, np.inf)
    
    return (hubble_time() / params.h) * I[0]
#end age
    
#-------------------------------------------------------------------------------
def Hubble_law(z, params=default_params):
    """
    Compute the comoving distance using Hubble's law, v_rec = H_0 * d, 
    where d is scale factor * comoving separation and z = v_rec/c
    [units: Mpc]
    
    Parameters
    ----------
    z : {float, array_like}
        the redshift to compute the distance to
    params : {str, dict, Cosmology}
        the cosmological parameters to use. Default is set by 
        the value of ``parameters.default_params``
    """
    if not isinstance(params, Cosmology):
        params = Cosmology(params)
        
    return z*(hubble_distance() / params.h)*(1+z)
#end Hubble_law
    
#-------------------------------------------------------------------------------
@np.vectorize
def Dc(z, params=default_params):
    """
    The line-of-sight comoving distance
    Remains constant with epoch if objects are in the Hubble flow
    
    Notes
    -----
    As given in Eq. 15 of Hogg 1999. It is equal to:
    
    .. math:: D_c = \int_0^z \frac{dz'}{E(z')}
    
    Parameters
    ----------
    z : {float, array_like}
        the redshift to compute the comoving distance at
    params : {str, dict, Cosmology}
        the cosmological parameters to use. Default is set by 
        the value of ``parameters.default_params``    
    """
    if z == 0:
        return 0.
    else:
        
        if not isinstance(params, Cosmology):
            params = Cosmology(params)
            
        f = lambda z: 1./E(z, params=params)
        I = integ.quad(f, 0, z)
        return (hubble_distance() / params.h) * I[0]
#end Dc
    
#-------------------------------------------------------------------------------
def Dm(z, params=default_params):
    """
    The transverse comoving distance. At the same redshift but separated by 
    an angle dtheta, Dm * dtheta is transverse comoving distance 
    [units: Mpc].
    
    Notes
    -----
    As given in Eq. 16 of Hogg 1999.
    
    Parameters
    ----------
    z : {float, array_like}
        the redshift to compute the distance at
    params : {str, dict, Cosmology}
        the cosmological parameters to use. Default is set by 
        the value of ``parameters.default_params``
    """
    if not isinstance(params, Cosmology):
        params = Cosmology(params)
        
    sOk = np.sqrt(np.abs(params.omegak))
    Dh = hubble_distance() / params.h
    if params.omegak < 0.:
        return Dh*np.sin(sOk * Dc(z, params=params)/Dh ) / sOk
    elif params.omegak == 0.:
        return Dc(z, params=params)
    else:
        return Dh * np.sinh(sOk * Dc(z, params=params)/Dh ) / sOk
#end Dm
    
#-------------------------------------------------------------------------------
def Da(z, params=default_params):
    """
    The angular diameter distance, which is equal to the ratio of an 
    object's physical transverse size to its angular size in radians
    [units: Mpc]
    
    Notes
    -----
    As given in Eq. 18 of Hogg 1999.
    
    Parameters
    ----------
    z : {float, array_like}
        the redshift to compute the distance at
    params : {str, dict, Cosmology}
        the cosmological parameters to use. Default is set by 
        the value of ``parameters.default_params``
    """
    return Dm(z, params=params) / (1.+ z)
#end Da
    
#-------------------------------------------------------------------------------
def Dl(z, params=default_params):
    """
    The luminosity distance [units: Mpc] 
    
    Notes
    -----
    As given in Eq. 21 of Hogg 1999.
    
    Parameters
    ----------
    z : {float, array_like}
        the redshift to compute the distance at
    params : {str, dict, Cosmology}
        the cosmological parameters to use. Default is set by 
        the value of ``parameters.default_params``
    """
    return (1. + z) * Dm(z, params=params)
#end Dl
    
#-------------------------------------------------------------------------------
@np.vectorize
def Dhor(z, params=default_params):
    """
    The horizon distance at redshift z, which is the comoving distance that 
    light can travel from z' = infty to z' = z [units: Mpc]
    
    Parameters
    ----------
    z : {float, array_like}
        the redshift to compute the distance at
    params : {str, dict, Cosmology}
        the cosmological parameters to use. Default is set by 
        the value of ``parameters.default_params``
    """
    if not isinstance(params, Cosmology):
        params = Cosmology(params)
        
    f = lambda zp : 1 / E(zp, params=params)
    I = integ.quad(f, z, np.inf)
    return (hubble_distance() / params.h) * I[0]
#end Dhor

#-------------------------------------------------------------------------------
def mu(z, params=default_params):
    """
    The distance modulus. [units: None]
    
    Notes
    -----
    As given in Eq. 25 of Hogg 1999.
        
    Parameters
    ----------
    z : {float, array_like}
        the redshift to compute the distance modulus at
    params : {str, dict, Cosmology}
        the cosmological parameters to use. Default is set by 
        the value of ``parameters.default_params``
    """
    return 5. * np.log10(Dl(z, params=params) * c.mega) - 5.
#end mu

#---------------------------------------------------------------------------
def dVc(z, params=default_params):
    """
    The differential comoving volume element :math: `dV_c / d\Omega / dz, 
    which has dimensions of volume per unit redshift per unit solid angle 
    [units :math: `Mpc^3 sr^{-1}`]
    
    Notes
    -----
    As given in Eq. 28 of Hogg 1999.
    
    Parameters
    ----------
    z : {float, array_like}
        the redshift to compute the quantity at
    params : {str, dict, Cosmology}
        the cosmological parameters to use. Default is set by 
        the value of ``parameters.default_params``
    """
    return (hubble_distance() / params.h) * Dm(z, params=params)**2. / E(z, params=params)
#end dVc
    
#-------------------------------------------------------------------------------
def Vc(z, params=default_params):
    """
    The comoving volume from z' = 0 to z' = z [units: Mpc] 
    
    Notes
    -----
    As given in Eq. 29 of Hogg 1999.
    
    Parameters
    ----------
    z : {float, array_like}
        the redshift to compute the quantity at
    params : {str, dict, Cosmology}
        the cosmological parameters to use. Default is set by 
        the value of ``parameters.default_params``
    """
    if not isinstance(params, Cosmology):
        params = Cosmology(params)
        
    Ok = params.omegak
    Dh = (hubble_distance() / params.h)
    sOk = np.sqrt(np.abs(Ok))
    Dm = Dm(z, params=params)
    if Ok < 0.:
        norm = 4*np.pi*Dh**3. / (2.*Ok)
        return norm * ( Dm/Dh*np.sqrt(1+Ok*(Dm/Dh)**2) - np.arcsin(sOk * Dm/Dh ) / sOk )
    elif Ok == 0.:
        return 4*np.pi*Dm**3 / 3.
    else:
        norm = 4*np.pi*Dh**3/(2.*Ok)
        return norm * ( Dm/Dh*np.sqrt(1+Ok*(Dm/Dh)**2) - np.arcsinh(sOk * Dm/Dh ) / sOk)
#end Vc

#-------------------------------------------------------------------------------
def angular_size(z, diameter, params=default_params):
    """
    The angular size of an object [units: degrees] of the input diameter
    [units: Mpc] at redshift z
    
    Parameters
    ----------
    z : float
        the redshift to compute angular size at 
    diameter :  float
        the physical diameter of the object in Mpc
    params : {str, dict, Cosmology}
        the cosmological parameters to use. Default is set by 
        the value of ``parameters.default_params``
    """
    return (diameter / Da(z, params=params)) / c.degree
#end angular_size

#-------------------------------------------------------------------------------
def physical_size(z, ang_size, params=default_params):
    """
    Compute the physical size [units: Mpc] corresponding to an angular size
    [units: degrees] at redshift z

    Parameters
    ----------
    z : float
        the redshift to compute sizes at
    ang_size : float
        the angular size, in degrees
    params : {str, dict, Cosmology}
        the cosmological parameters to use. Default is set by 
        the value of ``parameters.default_params``
    """
    return (ang_size*c.degree)*Da(z, params=params)
    #end physical_size
    
#-------------------------------------------------------------------------------
def lens_kernel_delta(z, z_s, params=default_params):
    """
    The value of the lensing kernel at a redshift z for a delta function
    distribution at source redshift z_s, assuming a flat universe [units: None]
    
    Parameters
    ----------
    z : {float, array_like}
        the redshift to compute the kernel at
    z_s : float
        the redshift of the source object
    params : {str, dict, Cosmology}
        the cosmological parameters to use. Default is set by 
        the value of ``parameters.default_params``
    """
    if not isinstance(params, Cosmology):
        params = Cosmology(params)
        
    D_s = Dm(z_s, params=params)
    D   = Dm(z, params=params)
    clight   = c.c_light / (c.km/c.second)
    
    cosmo_factor = 1.5*params.omegam*params.H0**2 * (1+z)/H(z, params=params)/clight
    geo_factor = D*(D_s-D)/D_s
    
    return cosmo_factor * geo_factor
#end lens_kernel

#-------------------------------------------------------------------------------
