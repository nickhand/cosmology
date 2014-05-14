cdef extern from "../include/halo_tools.h":
    
    void avg_bias_integral(double *k, int numk, double *x_spline, double *y_spline,
                           int n_spline,  double Rmin, double Rmax, double *output) nogil