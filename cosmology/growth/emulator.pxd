cdef extern from "emulator/core.h":
    
    void emulator_wrapper(double ombh2, double ommh2, double ns, double H0, double w, 
                            double sigma8, double z, int use_cmbh, double *ystar) nogil

cdef extern from "hod_emulator/core.h":
    
    void hod_emulator_wrapper(double M_cut, double M1, double sigma, double kappa, 
                                double alpha, double outputredshift, double *output_pk) nogil
    
cpdef nonlinear_power(object z, object k=*, bint use_cmbh=*, object params=*)   
