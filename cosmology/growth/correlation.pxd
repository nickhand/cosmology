cdef extern from "../include/power_tools.h":
    void correlation_integral(double *r, int numr, double *k, double *Pk, 
                              int numk, double kmin, double *output) nogil
    void avg_correlation_integral(double *r, int numr, double *r_spline, double *xi_spline,
                                  int n_spline,  double *output) nogil