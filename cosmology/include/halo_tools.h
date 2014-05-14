#ifndef __HALO_TOOLS_H
#define __HALO_TOOLS_H

void avg_bias_integral(double *k, int numk, double *x_spline, double *y_spline,
                       int n_spline,  double Rmin, double Rmax, double *output);

double bias_integrand(double R, void *params);


#endif