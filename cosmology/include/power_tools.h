#ifndef __POWER_TOOLS_H
#define __POWER_TOOLS_H

#include <gsl/gsl_spline.h>

#define N_a     (100)
#define N_k     (100)

#define a_min (0.0009) // this is roughly 1/(1+1100)
#define k_min (1.e-3)
#define k_max (1.e6)

/* global variables */
double  omega_m,		/* Density of CDM and baryons, in units of critical density */
        omega_b,        /* Density of baryons, in units of critical density */
        omega_l,		/* Density of dark energy in units of critical density */
        omega_r,        /* Density of radiation in units of critical density */
        omega_k,        /* Curvature parameter */
        sigma_8,		/* Mass variance on scale of R = 8 Mpc/h */
        n_spec, 		/* Primordial power spectrum spectral index */
        hubble,         /* Hubble parameter in units of 100 km/s/Mpc */
        Tcmb,           /* Temperature of the CMB*/
        w_lam,          /* Dark energy equation of state */
        f_baryon;       /* Baryon fraction */
        
int  transfer;          /* which transfer function to use */

/* spline transfer */
gsl_spline * transfer_spline;
gsl_interp_accel * transfer_acc;

// the main callable cosmology functions
void set_parameters(double OMEGAM, double OMEGAB, double OMEGAL, double OMEGAR, 
                    double SIGMA8, double HUBBLE, double NSPEC, double TCMB, 
                    double W_LAM, int TRANSFER);
void set_CAMB_transfer(double *k, double *Tk, int numk);
void free_transfer(void);
double E(double a);
void D_plus(double *z, int numz, int normed, double *growth);
void unnormalized_sigma_r(double *r, double z, int numr, double *sigma);
void unnormalized_transfer(double *k, double z, int numk, double *transfer);
double normalize_power(void);
void nonlinear_power(double *k, double z, int numk, double *power);
void dlnsdlnm_integral(double *r, int numr, double *output);
double omegal_a(double a);
double omegam_a(double a);
double Delta_L_onek(double k);

void correlation_integral(double *r, int numr, double *k, double *Pk, int numk,  
                          double kmin, double *output);
void avg_correlation_integral(double *r, int numr, double *r_spline, double *xi_spline,
                              int n_spline,  double *output);
void lens_kern_integral(double *z, int numz, double *z_spline, double *nz,
                        double *Dm, int n_spline, double zmax, double *output);
void lens_power_integral(double *ell, int numl, double *z_spline, double *integrand,
                         double *Dm, int n_spline, double zmin, double zmax, 
                         int linear, double *output);   
    
// these are internal helper functions for the main cosmology ones
double growth_function_integrand(double a, void *params);
double sigma2_integrand(double k, void *params);
double dlnsdlnm_integrand(double k, void *params);
double unnormalized_transfer_onek(double k);
double TF_spline(double k);
double correlation_integrand(double k, void *params);
double avg_corr_integrand(double r, void *params);
double lens_kern_integrand(double lnzs, void *params);
double lens_power_integrand(double lnz, void *params);

// halofit functions
void halofit(double rk, double rn, double rncur, double rknl, double plin, 
    double om_m, double om_v, double *pnl);
double int_for_wint2_ncur(double logk);
double int_for_wint2_neff(double logk);
double int_for_wint2_knl(double logk);
void wint2(double r, double *sig, double *d1, double *d2, double amp, int onlysig);
double slope_NL(double rn, double rncur, double om_m, double om_v);

//  numerical recipes functions used during the nonlinear power calculation
double dlog(double x);
double qromb1(double (*func)(double), double a, double b);
double trapzd1(double (*func)(double), double a, double b, int n);
void error(const char *s);
void polint(double xa[], double ya[], int n, double x, double *y, double *dy);
void free_vector(double *v, long nl, long nh);
double *vector(long nl, long nh);
double **matrix(long nrl, long nrh, long ncl, long nch);
void spline(double x[], double y[], int n, double yp1, double ypn, double y2[]);
void splint(double xa[], double ya[], double y2a[], int n, double x, double *y);
double interpol(double *f, int n, double a, double b, double dx, double x,
                double lower, double upper);
double interpol2d(double **f,
		          int nx, double ax, double bx, double dx, double x,
		          int ny, double ay, double by, double dy, double y,
                  double lower, double upper);
	            
#define NR_END 1
#define FREE_ARG char*

#endif