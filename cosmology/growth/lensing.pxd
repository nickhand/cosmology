cdef extern from "../include/power_tools.h":
    void lens_kern_integral(double *z, int numz, double *z_spline, double *nz,
                            double *Dm, int n_spline, double zmax, double *output) nogil
                            
    void lens_power_integral(double *ell, int numl, double *z_spline, double *integrand,
                             double *Dm, int n_spline, double zmin, double zmax, 
                             bint linear, double *output) nogil
    void set_parameters(double OMEGAM, double OMEGAB, double OMEGAL, double OMEGAR, 
                        double SIGMA8, double HUBBLE, double NSPEC, double TCMB, 
                        double W_LAM, int TRANSFER) nogil
    void set_CAMB_transfer(double *k, double *Tk, int numk) nogil