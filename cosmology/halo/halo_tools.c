#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sys.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>
#include "../include/halo_tools.h"

// define the function parameterss
typedef struct { 
    double k;
    gsl_spline *spline;
    gsl_interp_accel *acc;
} fparams;

/*----------------------------------------------------------------------------*/

double bias_integrand(double R, void *params)
/*
    The integrand that calls the spline function
*/
{
    double spline;
    fparams p = *((fparams *) params);
    
    double x = p.k*R;
    double W  = 3.*(sin(x) - x*cos(x))/(x*x*x);
    
    // get the spline values 
    spline = gsl_spline_eval(p.spline, R, p.acc);
    if (gsl_isnan(spline)) {
        spline = 0.;
    }
        
    return W*W*spline*R*R;
}
/*----------------------------------------------------------------------------*/
void avg_bias_integral(double *k, int numk, double *x_spline, double *y_spline,
                       int n_spline,  double Rmin, double Rmax, double *output)
/*
    Compute the integral needed to compute the average halo bias
*/
{
    int i;
    double result, error;
    
    // spline variables
    gsl_spline *spline;
    gsl_interp_accel *acc;
    
    // initialize the spline
    spline = gsl_spline_alloc(gsl_interp_cspline, n_spline);
    acc    = gsl_interp_accel_alloc();
    gsl_spline_init(spline, x_spline, y_spline, n_spline);
    
    // set up the gsl function and integration workspace
    gsl_integration_cquad_workspace * work = gsl_integration_cquad_workspace_alloc(1000);
    
    // set up the function params
    fparams params;
    params.spline = spline;
    params.acc    = acc;
    
    // set up the gsl function 
    gsl_function F;
    F.function = &bias_integrand;

    
    // integrate for every radius provided    
    for (i=0; i < numk; i++) {
           
        params.k = k[i];
        F.params = &params; 
        
        gsl_integration_cquad(&F, Rmin, Rmax, 0., 1e-4, work, &result, &error, NULL);
        output[i] = result;
    }
    
    // free the integration workspace
    gsl_integration_cquad_workspace_free(work);
    gsl_spline_free(spline);
    gsl_interp_accel_free(acc);
}