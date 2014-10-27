#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sys.h>
#include <gsl/gsl_errno.h>
#include "../include/power_tools.h"
#include "../include/transfer.h"

// define structs to pass parameters for integrating
typedef struct { 
    double r;
    gsl_spline *spline;
    gsl_interp_accel *acc;
} fparams;

typedef struct { 
    double x; 
    gsl_spline *spline1;
    gsl_interp_accel *acc1;
    gsl_spline *spline2;
    gsl_interp_accel *acc2;
    int linear; 
} lens_params;

double power_norm;

/*----------------------------------------------------------------------------*/
void set_parameters(double OMEGAM, double OMEGAB, double OMEGAL, double OMEGAR, 
                    double SIGMA8, double HUBBLE, double NSPEC, double TCMB, 
                    double W_LAM, int TRANSFER)
/* 
    Set the parameters needed for power spectrum calculations 
    
    Input:  OMEGAM -- Density of CDM and baryons, in units of critical density 
            OMEGAB -- Density of baryons, in units of critical density 
            OMEGAV -- Density of dark energy in units of critical density
            OMEGAR -- Density of radiation, in units of critical density
            SIGMA8 -- Mass variance on scale of R = 8 Mpc/h 
            HUBBLE -- Hubble parameter in units of 100 km/s/Mpc 
            NSPEC -- Primordial power spectrum spectral index 
            TCMB -- Temperature of the CMB,
            W_LAM -- Dark energy equation of state
            TRANSFER -- The integer specifying which transfer function to use.
*/
{
    omega_m  = OMEGAM;
    omega_b  = OMEGAB;
    omega_l  = OMEGAL;
    omega_r  = OMEGAR;
    omega_k  = 1. - omega_m - omega_l - omega_r;
    sigma_8  = SIGMA8;
    n_spec   = NSPEC; 
    hubble   = HUBBLE;
    Tcmb     = TCMB;
    w_lam    = W_LAM;
    f_baryon = omega_b/omega_m; 
    if (TRANSFER >= 0) {
        transfer = TRANSFER; 
        
        transfer_spline = NULL;
        transfer_acc = NULL;
    }
        
    if ((transfer == 0) || (transfer == 1) || (transfer == 2)) {
        TF_eh_set_parameters(omega_m*hubble*hubble, f_baryon, Tcmb);
    }
    
    power_norm = 0.;
    gsl_set_error_handler_off();
}
/*----------------------------------------------------------------------------*/
void free_transfer(void) 
/*
    Utility function to free the transfer splines
*/
{
    if (transfer_spline != NULL) {
        gsl_spline_free(transfer_spline);
    } 
    if (transfer_acc != NULL) {
        gsl_interp_accel_free(transfer_acc);
    }
}

/*----------------------------------------------------------------------------*/
void set_CAMB_transfer(double *k, double *Tk, int numk)
/*
    Initialize the transfer function parameters and normalize the power
*/
{
    free_transfer();

    transfer_spline = gsl_spline_alloc(gsl_interp_cspline, numk); 
    transfer_acc = gsl_interp_accel_alloc();  
    gsl_spline_init(transfer_spline, k, Tk, numk);

}
/*----------------------------------------------------------------------------*/
double TF_spline(double k)
/*
    Compute the transfer function given an input spline function
    
    Input: k -- wavenumber in h/Mpc
*/
{
    double Tk = gsl_spline_eval(transfer_spline, k, transfer_acc);
    
    /* default to full EH transfer function if error*/
    if (gsl_isnan(Tk)) {
        Tk = 0.;
    }
    return Tk;
}

/*----------------------------------------------------------------------------*/
double E(double a) {
/*
    The dimensionaless Hubble parameter, H(z) / H0
    
    Input: a -- Scale factor 
*/
    if (a == 0) {
        return 0.;
    } else {
        return sqrt(omega_m/(a*a*a) + omega_r/(a*a*a*a) + omega_k/(a*a) + omega_l/pow(a, 3.*(1+w_lam)));
    }
}

/*----------------------------------------------------------------------------*/
double normalize_power(void)
/*
    Normalize the power spectrum to the value of sigma_8 at z = 0 at R = 8 h/Mpc
*/
{
    double sigma0, rnorm = 8.;
    unnormalized_sigma_r(&rnorm, 0., 1, &sigma0);
    return sigma_8*sigma_8 / (sigma0*sigma0);
}

/*----------------------------------------------------------------------------*/
double growth_function_integrand(double a, void *params)
/*
    Integrand used internally in D_plus() to compute the growth function
*/
{
    double integral, Ea;
        
    if (a == 0.) {
        integral = 0.;
    } else {
        Ea = E(a);
        integral = 1./(Ea*Ea*Ea*a*a*a);   
    }
    return integral;
}
/*----------------------------------------------------------------------------*/
void D_plus(double *z, int numz, int normed, double *growth)
/*
    The linear growth function, normalized to unity at z = 0, if normed=1
    
    Input:  z -- the array of redshifts to compute the function at
            numz -- the number of redshifts to compute at
            normed -- whether to normalize to unity at z = 0
            growth -- the growth function values to return
*/
{
    int i;
    double result, error, Ea, norm;
    
    // set up the gsl function and integration workspace
    gsl_integration_cquad_workspace * work = gsl_integration_cquad_workspace_alloc(1000);
    gsl_function F;
    F.function = &growth_function_integrand;
    F.params = NULL;

    // compute the normalization
    if (normed) {
        double this_z = 0.;
        D_plus(&this_z, 1, 0, &norm);
    } else {
        norm = 1.;
    }

    
    // integrate for every redshift provided    
    for (i=0; i < numz; i++) {
        gsl_integration_cquad(&F, 0., 1./(1.+z[i]), 0., 1e-5, work, &result, &error, NULL);
        
        Ea = E(1./(1.+z[i]));
        growth[i] = 5./2.*omega_m*Ea*result/norm;
    }
    // free the integration workspace
    gsl_integration_cquad_workspace_free(work);
}
/*----------------------------------------------------------------------------*/
double sigma2_integrand(double k, void *params)
/*
    Integrand of the sigma squared integral, used internally by the 
    sigma_r() function. It is equal to k^2 W(kr)^2 T(k)^2 k^n_spec, 
    where the transfer function used is specified by the global 
    parameter ``transfer``.
*/
{
    double r  = *((double*)params);
    double W  = 3.*(sin(k*r)-k*r*cos(k*r))/(k*k*k*r*r*r);
    double Tk = unnormalized_transfer_onek(k);
    
    return (k*k*W*W)*(Tk*Tk*pow(k, n_spec))/(2.*M_PI*M_PI);
}
/*----------------------------------------------------------------------------*/
void unnormalized_sigma_r(double *r, double z, int numr, double *sigma)
/*
    The unnormalized verage mass fluctuation within a sphere of radius r, using the specified
    transfer function for the linear power spectrum.
    
    Input: r -- radii to compute the statistic at in Mpc/h
           z -- the redshift to compute at
           numr -- the number of radii for which to compute sigma
           normed -- whether to normalize to sigma_8 at z = 0
           sigma -- the output statistic         
*/
{
    int i;
    double result, error;
    
    // set up the gsl function and integration workspace
    gsl_integration_cquad_workspace * work = gsl_integration_cquad_workspace_alloc(1000);
    gsl_function F;
    F.function = &sigma2_integrand;
    
    // integrate for every radius provided    
    for (i=0; i < numr; i++) {
        F.params = &r[i];
        gsl_integration_cquad(&F, 1e-3/r[i], 100./r[i], 0., 1e-5, work, &result, &error, NULL);
        sigma[i] = sqrt(result);
    }
    
    // free the integration workspace
    gsl_integration_cquad_workspace_free(work);
}
/*----------------------------------------------------------------------------*/
void unnormalized_transfer(double *k, double z, int numk, double *transfer)
/*
    The unnormalized transfer function specified by the ``transfer`` global parameter.
    
    Input: k -- wavenumber to compute power spectrum at in h/Mpc
           z -- the redshift to compute at
           numk -- the number of wavenumber for which to compute power
           power -- the output power spectrum
*/
{
    int i;
    for (i=0; i < numk; i++){
        transfer[i] = unnormalized_transfer_onek(k[i]);
    }
}
/*----------------------------------------------------------------------------*/
double dlnsdlnm_integrand(double k, void *params)
/*
    Integrand for the dln(sigma)/dln(mass) integral, for use internally
    by the dlnsdlnm_integral() function.
*/
{

    double r  = *((double*)params);
    double Tk, dW2dm; 
    
    Tk = unnormalized_transfer_onek(k);
    dW2dm = (sin(k*r)-k*r*cos(k*r))*(sin(k*r)*(1.-3./(k*k*r*r))+3*cos(k*r)/(k*r));
    return dW2dm*(Tk*Tk*pow(k, n_spec))/(k*k);
}
/*----------------------------------------------------------------------------*/
void dlnsdlnm_integral(double *r, int numr, double *output)
/*
    Helper function for computing the integral in dln(sigma)/dln(mass). It is
    used to computed halo mass functions and is given by \int dW^2/dM P(k) / k^2 dk.
    Note that it is computed at z = 0. The power integral is unnormalized.
    
    Input: r -- radii to compute the statistic at in Mpc/h
           numr -- the number of radii for which to compute sigma
           output -- the output statistic
*/
{
    int i;
    double result, error;
    
    // set up the gsl function and integration workspace
    gsl_integration_cquad_workspace * work = gsl_integration_cquad_workspace_alloc(1000);
    gsl_function F;
    F.function = &dlnsdlnm_integrand;
    
    // integrate for every radius provided    
    for (i=0; i < numr; i++) {
        F.params = &r[i];
        gsl_integration_cquad(&F, 1e-3/r[i], 100./r[i], 0., 1e-5, work, &result, &error, NULL);
        output[i] = result;
    }
    
    // free the integration workspace
    gsl_integration_cquad_workspace_free(work);
}
/*----------------------------------------------------------------------------*/
double unnormalized_transfer_onek(double k)
/*
    Helper function to compute the unnormalize linear power for one k (in h/Mpc),
    given by T(k)^2 * k^n_spec
*/
{ 
    double Tk;
    
    if (transfer == 0) {
        Tk = TF_eh_onek(k*hubble);
    } else if (transfer == 1) {
        Tk = TF_eh_nowiggles(omega_m, f_baryon, hubble, Tcmb, k);
    } else if (transfer == 2) {
        Tk = TF_eh_zerobaryon(omega_m, hubble, Tcmb, k);
    } else if (transfer == 3) {
        Tk = TF_bbks(omega_m, f_baryon, hubble, k);
    } else if (transfer == 4) {
        Tk = TF_bond_efs(omega_m, f_baryon, hubble, k);
    } else if (transfer == 5) {
        Tk = TF_spline(k);
    }   
    return Tk;
}
/*----------------------------------------------------------------------------*/
double omegal_a(double a) 
/*
    The dark energy density omega_l as a function of scale factor.
*/
{   double Ea = E(a);
    return omega_l / (Ea*Ea);
}

/*----------------------------------------------------------------------------*/
double omegam_a(double a)
/*
    The matter density omega_m as a function of scale factor.
*/
{
    double Ea = E(a);
    return omega_m / (Ea*Ea*a*a*a);
}
/*----------------------------------------------------------------------------*/
double correlation_integrand(double k, void *params)
/*
   Integrand of the correlation function integral, used internally by the 
   correlation_integral() function. It is equal to k^2 P(k) sin(kR) / (kR).
   We are integrating over wavenumbers k here.
*/
{

    double power;
    fparams p = *((fparams *) params);

    // get the spline values 
    power = gsl_spline_eval(p.spline, k, p.acc);
    if (gsl_isnan(power)) {
        power = 0.;
    }
        
    return k*power/p.r;
}
/*----------------------------------------------------------------------------*/
void correlation_integral(double *r, int numr, double *k, double *Pk, int numk,  
                          double kmin, double *output)
/*
    Compute the correlation function at input radii in Mpc/h
    
    
    Input: r -- radii to compute the statistic at in Mpc/h
           numr -- the number of radii for which to compute the correlation
           k -- wavenumbers in h/Mpc for use in the spline of the power spectrum
           Pk -- power spectrum in (Mpc/h)^3 for use in spline
           numk -- the number of spline points
           kmin -- the minimum wavenumber to integrate over
           output -- the output statistic
*/
{
    int i;
    double result, error;
    
    // spline variables
    gsl_spline *spline;
    gsl_interp_accel *acc;
    
    // integration variables
    gsl_integration_workspace *w;
    gsl_integration_workspace *w_cycle;
    gsl_integration_qawo_table *integ_table;
    
    // initialize the spline
    spline = gsl_spline_alloc(gsl_interp_cspline, numk);
    acc    = gsl_interp_accel_alloc();
    gsl_spline_init(spline, k, Pk, numk);

    // set up the integration workspace and tables
    integ_table = gsl_integration_qawo_table_alloc(1., 1., GSL_INTEG_SINE, 1000);
    w           = gsl_integration_workspace_alloc(1000);
    w_cycle     = gsl_integration_workspace_alloc(1000);
    
    // set up the function params
    fparams params;
    params.spline = spline;
    params.acc    = acc;
    
    // set up the gsl function 
    gsl_function F;
    F.function = &correlation_integrand;
    
    // integrate for every radius provided    
    for (i=0; i < numr; i++) {
        
        params.r = r[i];
        F.params = &params;
        
        gsl_integration_qawo_table_set(integ_table, r[i], 1., GSL_INTEG_SINE);
        gsl_integration_qawf(&F, kmin, 1e-4, 1000, w, w_cycle, integ_table, &result, &error);
    
        output[i] = result / (2.*M_PI*M_PI);
    }
    
    // free everything 
    gsl_spline_free(spline);
    gsl_interp_accel_free(acc);
    gsl_integration_qawo_table_free(integ_table);
    gsl_integration_workspace_free(w);
    gsl_integration_workspace_free(w_cycle);
}
/*----------------------------------------------------------------------------*/
double avg_corr_integrand(double r, void *params)
/*
    Integrand of the correlation function integral, averaged over a sphere 
    of radius R, used internally by the avg_correlation_integral() function. 
    It is equal to 3*r^2*xi(r). We are integrating over radii r here.
*/
{
    double xi;
    fparams p = *((fparams *) params);

    // get the spline values 
    xi = gsl_spline_eval(p.spline, r, p.acc);
    if (gsl_isnan(xi)) {
        xi = 0.;
    }
        
    return 3*r*r*xi;
}
/*----------------------------------------------------------------------------*/
void avg_correlation_integral(double *r, int numr, double *r_spline, double *xi_spline,
                              int n_spline,  double *output)
/*
    Compute the correlation function, averaged over a sphere of radius R 
    in Mpc/h. It is given by: xi(y) = 3/y^3 \int x^2 xi(x)


    Input: r -- radii to compute the statistic at in Mpc/h
           numr -- the number of radii for which to compute the correlation
           r_spline -- radii in Mpc/h for use in the spline 
           xi_spline -- correlation function for use in spline
           n_spline -- the number of spline points
           output -- the output statistic
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
    gsl_spline_init(spline, r_spline, xi_spline, n_spline);
    
    // set up the gsl function and integration workspace
    gsl_integration_cquad_workspace * work = gsl_integration_cquad_workspace_alloc(1000);
    
    // set up the function params
    fparams params;
    params.spline = spline;
    params.acc    = acc;
    params.r      = 0.;
    
    // set up the gsl function 
    gsl_function F;
    F.function = &avg_corr_integrand;
    F.params = &params;
    
    // integrate for every radius provided    
    for (i=0; i < numr; i++) {
            
        gsl_integration_cquad(&F, 0., r[i], 0., 1e-4, work, &result, &error, NULL);
        output[i] = result / (r[i]*r[i]*r[i]);
    }
    
    // free the integration workspace
    gsl_integration_cquad_workspace_free(work);
    gsl_spline_free(spline);
    gsl_interp_accel_free(acc);
}
/*----------------------------------------------------------------------------*/
double lens_kern_integrand(double lnz, void *params)
/*
    Integrand to compute the lensing kernel, used internally by the 
    lens_kern_integral() function. We are integrating over log(z) here. 
*/
{
    double Dm_x, Dm_z, nz;
    lens_params p = *((lens_params *) params);
    double z = exp(lnz);
    
    // get the spline values 
    nz = gsl_spline_eval(p.spline1, z, p.acc1);
    if (gsl_isnan(nz)) {
        nz = 0.;
    }
    Dm_z = gsl_spline_eval(p.spline2, z, p.acc2);
    if (gsl_isnan(Dm_z)) {
        Dm_z = 0.;
    }
    Dm_x = gsl_spline_eval(p.spline2, p.x, p.acc2);
    if (gsl_isnan(Dm_x)) {
        Dm_x = 0.;
    }   
    return z*nz*(1. - Dm_x/Dm_z);
}
/*----------------------------------------------------------------------------*/
void lens_kern_integral(double *z, int numz, double *z_spline, double *nz,
                        double *Dm, int n_spline, double zmax, double *output)
/*
    Compute the lensing kernel integral, for use in computing a lensing power
    spectrum. It is given by \int_z^\infty dx n(x) (1 - Dm(z)/Dm(x))


    Input: z -- redshift to compute the kernel at
           numz -- the number of redshifts
           z_spline -- redshifts where the splines will be defined 
           nz -- the redshift distribution for use in spline
           Dm -- the comoving distance for use in spline
           n_spline -- the number of spline points
           zmax -- the maximum redshift to integrate out to
           output -- the output statistic  
*/
{
    int i;
    double result, error;
    
    // set up the n(z) spline variables
    gsl_spline *nz_spline;
    gsl_interp_accel *nz_acc;
    
    // and the Dm(z) spline vars
    gsl_spline *Dm_spline;
    gsl_interp_accel *Dm_acc;
    
    // initialize the n(z) spline
    nz_spline = gsl_spline_alloc(gsl_interp_cspline, n_spline);
    nz_acc    = gsl_interp_accel_alloc();
    gsl_spline_init(nz_spline, z_spline, nz, n_spline);
    
    // initialize the Dm(z) spline
    Dm_spline = gsl_spline_alloc(gsl_interp_cspline, n_spline);
    Dm_acc    = gsl_interp_accel_alloc();
    gsl_spline_init(Dm_spline, z_spline, Dm, n_spline);
    
    // set up the gsl function and integration workspace
    gsl_integration_cquad_workspace * work = gsl_integration_cquad_workspace_alloc(1000);
    
    // set up the function params
    lens_params params;
    params.spline1 = nz_spline;
    params.acc1    = nz_acc;
    params.spline2 = Dm_spline;
    params.acc2    = Dm_acc;
    
    // set up the gsl function 
    gsl_function F;
    F.function = &lens_kern_integrand;
    
    // integrate for every radius provided    
    for (i=0; i < numz; i++) {
        
        params.x = z[i];
        F.params = &params;
        
        if (z[i] >= zmax) {
            output[i] = 0.;
        } else {
            gsl_integration_cquad(&F, log(z[i]), log(zmax), 0., 1e-4, work, &result, &error, NULL);        
            output[i] = result;
        }
    }
    
    // free the integration workspace
    gsl_integration_cquad_workspace_free(work);
    
    // free the splines
    gsl_spline_free(nz_spline);
    gsl_interp_accel_free(nz_acc);
    
    gsl_spline_free(Dm_spline);
    gsl_interp_accel_free(Dm_acc);

}

/*----------------------------------------------------------------------------*/
double lens_power_integrand(double lnz, void *params)
/*
    Integrand to compute the lensing power spectrum, used internally by the 
    lens_power_integral() function. We are integrating over log(z) here.
*/
{
    double Dm, integ, k;
    double Pk_L, D2_L, Pk_NL;
    double growth;
    lens_params p = *((lens_params *) params);
    double z = exp(lnz);
    
    // get the spline values 
    integ = gsl_spline_eval(p.spline1, z, p.acc1);
    if (gsl_isnan(integ)) {
        integ = 0.;
    }
    Dm = gsl_spline_eval(p.spline2, z, p.acc2);
    if (gsl_isnan(Dm)) {
        k = 0.;
    } else {
        k = p.x / Dm;
    }
        
    // determine the power value at this k
    if (p.linear == 1){    
        
        // the growth function
        D_plus(&z, 1, 0, &growth);
        Pk_L = (growth*growth)*Delta_L_onek(k)*(2.*M_PI*M_PI)/(k*k*k);
        return z*integ*Pk_L; 
    }
    else {
        nonlinear_power(&k, z, 1, &Pk_NL);
        return z*integ*Pk_NL;
    }
}
/*----------------------------------------------------------------------------*/
void lens_power_integral(double *ell, int numl, double *z_spline, double *integrand,
                         double *Dm, int n_spline, double zmin, double zmax, 
                         int linear, double *output)
/*  
    Compute the lensing power spectrum integral. It is given by
    \int dz integrand(z) * Pk( k = ell/Dm, z)


    Input: ell -- multipole numbers to compute the spectrum at
           numl -- the number of multipoles
           z_spline -- redshifts where the splines will be defined 
           integrand -- the main integrand at a given redshift for use in spline
           Dm -- the comoving distance for use in spline
           n_spline -- the number of spline points
           zmin -- the minimum redshift to integrate over
           zmax -- the maximum redshift to integrate out to
           linear -- whether to use the linear or nonlinear power spectrum
           output -- the output statistic
*/
{
    int i;
    double result, error, growth_norm;
    
    // set up the integrand spline variables
    gsl_spline *integ_spline;
    gsl_interp_accel *integ_acc;
    
    // and the Dm(z) spline vars
    gsl_spline *Dm_spline;
    gsl_interp_accel *Dm_acc;
    
    // initialize the integrand spline
    integ_spline = gsl_spline_alloc(gsl_interp_cspline, n_spline);
    integ_acc    = gsl_interp_accel_alloc();
    gsl_spline_init(integ_spline, z_spline, integrand, n_spline);
    
    // initialize the Dm(z) spline
    Dm_spline = gsl_spline_alloc(gsl_interp_cspline, n_spline);
    Dm_acc    = gsl_interp_accel_alloc();
    gsl_spline_init(Dm_spline, z_spline, Dm, n_spline);
    
    // set up the gsl function and integration workspace
    gsl_integration_cquad_workspace * work = gsl_integration_cquad_workspace_alloc(1000);
    
    // set up the function params
    lens_params params;
    params.spline1 = integ_spline;
    params.acc1    = integ_acc;
    params.spline2 = Dm_spline;
    params.acc2    = Dm_acc;
    params.linear  = linear;
    
    // set up the gsl function 
    gsl_function F;
    F.function = &lens_power_integrand;
    
    // set the global power norm variable
    power_norm = normalize_power();
    double znorm = 0.;
    D_plus(&znorm, 1, 0, &growth_norm);
    
    // integrate for every ell provided    
    for (i=0; i < numl; i++) {
        
        params.x = ell[i];
        F.params = &params;
        
        gsl_integration_cquad(&F, log(zmin), log(zmax), 0., 1e-4, work, &result, &error, NULL);        
        if (linear == 1){
            result = result / (growth_norm*growth_norm);
        }
        output[i] = result;
    }
    
    // free the integration workspace
    gsl_integration_cquad_workspace_free(work);
    
    // free the splines
    gsl_spline_free(integ_spline);
    gsl_interp_accel_free(integ_acc);
    
    gsl_spline_free(Dm_spline);
    gsl_interp_accel_free(Dm_acc);

}
/*----------------------------------------------------------------------------*/
double Delta_L_onek(double k)
/*
    Return the dimensionaless linear power spectrum at input wavenumber k, with
    units of h/Mpc.
*/
{
 
    double Tk, D2_L; 
    Tk = unnormalized_transfer_onek(k);
    if (power_norm == 0.)
        power_norm = normalize_power();
    D2_L = (power_norm*(Tk*Tk*pow(k, n_spec))*k*k*k)/(2.*M_PI*M_PI);
    
    return D2_L;
}
/*----------------------------------------------------------------------------*/

void nonlinear_power(double *k, double z, int numk, double *power)
/*
    The nonlinear matter power spectrum, using the Halofit prescription. This
    code is adapted from Martin Kilbinger's Halofit+ code in C, which was based
    on the original Halofit code by Rob Smith.
    
    Input: k -- wavenumber to compute power spectrum at in h/Mpc
           z -- the redshift to compute at
           numk -- the number of wavenumber for which to compute power
           power -- the output power spectrum
*/
{
    static double OMEGA_M   = -42.;
    static double OMEGA_B   = -42.;
    static double OMEGA_L   = -42.;
    static double N_SPEC    = -42.;
    static double SIGMA_8   = -42.;
    static int TRANSFER     = -42;

    static double upper;
    static double table_k[N_k], table_P[N_k], y2[N_k], table_slope[N_a];
    static double **table_P_NL = 0;
    static double logkmin = 0., logkmax = 0., dk = 0., da = 0.;
    
    const double k_nonlin_max = 1.e6;   
    const int itermax  = 20;
    const double logstep = 5.0;
    
    double a, aa, zz, klog, amp, omv, omm, val;
    double logr1, logr2, logr1start, logr2start, logrmid, logrmidtmp;
    double rmid, sig, diff, d1, d2; 
    double rknl, rneff, rncur;
    double k_L, Delta_L, Delta_NL, lnk_NL;
    int i, j, golinear, iter; 
       
    if (z == 0.) {
        a = 0.99999;
    } else {
        a = 1./(1.+z);
    }
    
    if (OMEGA_M != omega_m || OMEGA_L != omega_l || N_SPEC != n_spec || 
        SIGMA_8 != sigma_8 || TRANSFER != transfer || OMEGA_B != omega_b) {

          if (!table_P_NL) table_P_NL = matrix(0, N_a-1, 0, N_k-1);

          /* upper = (dlnP/dlnk)_{k=kmax}, for splines & extrapolation */
          /* Note that the in the range considered here the linear power
           * spectrum is still a bit shallower than k^(n-4) since T(k) has
           * not yet reached its asymptotic limit. 
           */

          da = (1. - a_min)/(N_a-1.);
          aa = a_min;
          logkmin = log(k_min);
          logkmax = log(k_max);
          dk = (logkmax - logkmin)/(N_k-1.);

          for (i=0; i < N_a; i++, aa +=da) {
              
              klog = logkmin;
              
              zz = 1./aa - 1.;
    	      D_plus(&zz, 1, 1, &amp);
    	      omm = omegam_a(aa);
              omv = omegal_a(aa);
    	      golinear = 0;
              
    	      /* find non-linear scale with iterative bisection */
    	      logr1 = -2.0;
    	      logr2 =  3.5;

    	      iterstart:

    	      logr1start = logr1;
    	      logr2start = logr2;

    	      iter = 0;
    	      do {
    	          logrmid = 0.5*(logr2+logr1);
    	          rmid    = pow(10, logrmid);
    	          wint2(rmid, &sig, 0x0, 0x0, amp, 1);

    	          diff = sig - 1.0;

    	          if (diff > 0.001)
    		          logr1 = dlog(rmid);
    	          if (diff < -0.001)
    		          logr2 = dlog(rmid);

    	      } while (fabs(diff) >= 0.001 && ++iter < itermax);

    	      if (iter >= itermax) {
    	          logrmidtmp = (logr2start+logr1start)/2.0;
    	       
    	          if (logrmid < logrmidtmp) {
    		          logr1 = logr1start-logstep;
    		          logr2 = logrmid;
    	          } else if (logrmid >= logrmidtmp) {
    		          logr1 = logrmid;
    		          logr2 = logr2start+logstep;
    	          }

    	       /* non-linear scale far beyond maximum scale: set flag golinear */
    	       double o = 1./pow(10, logr2);
    	       if (1./pow(10, logr2) > k_nonlin_max) {
    		       golinear = 1;
    		       upper = table_slope[i] = n_spec-4.0;
    		       goto after_wint;
    	        } else {
    		        goto iterstart;
    	        }
    	    }

    	    /* spectral index & curvature at non-linear scale */
    	    wint2(rmid, &sig, &d1, &d2, amp, 0);
    	    rknl  = 1./rmid;
    	    rneff = -3-d1;
    	    rncur = -d2;

    	    upper = table_slope[i] = slope_NL(rneff, rncur, omm, omv);
	    
            after_wint:

            for (j=0; j<N_k; j++, klog+=dk) {
    	    
    	        k_L = exp(klog);

	            Delta_L = amp*amp*Delta_L_onek(k_L);
	            
	            if (golinear == 0) {
		            halofit(k_L, rneff, rncur, rknl, Delta_L, omm, omv, &Delta_NL);
	            } else {
		            Delta_NL = Delta_L;
	            }
                
	            lnk_NL = klog;

    	        table_k[j] = lnk_NL;
    	        table_P[j] = log(2*M_PI*M_PI*Delta_NL) - 3.*lnk_NL; /* PD (3) */
            }

    	    spline(table_k-1, table_P-1, N_k, n_spec, upper, y2-1);
            klog = logkmin;
            for (j=0; j<N_k; j++, klog += dk) {
                splint(table_k-1, table_P-1, y2-1, N_k, klog, &val);
    	        table_P_NL[i][j] = val;
            }
        }

        OMEGA_M = omega_m;
        OMEGA_L = omega_l;
        N_SPEC  = n_spec;
        SIGMA_8 = sigma_8;
        OMEGA_B = omega_b;
        TRANSFER = transfer;
    }

    for (i=0; i < numk; i++){
        
        klog = log(k[i]);

        upper = interpol(table_slope, N_a, a_min, 1.0, da, a, 1e31, 1e31);
        val = interpol2d(table_P_NL, N_a, a_min, 1., da, a, N_k, logkmin, logkmax, 
		                dk, klog, n_spec, upper);
        power[i] = exp(val);
    }
}
    
/*----------------------------------------------------------------------------*/
void halofit(double rk, double rn, double rncur, double rknl, double plin, 
      double om_m, double om_v, double *pnl)
/*
    Halo model nonlinear fitting formula as described in Appendix C of 
    Smith et al. 2003.
    
    The halofit in Smith et al. 2003 predicts a smaller power
    than latest N-body simulations at small scales.
    Update the following fitting parameters of gam,a,b,c,xmu,xnu,
    alpha & beta from the simulations in Takahashi et al. 2012.
    The improved halofit accurately provide the power spectra for WMAP
    cosmological models with constant w.
*/
{
    double gam,a,b,c,xmu,xnu,alpha,beta,f1,f2,f3;
    double y, ysqr;
    double f1a,f2a,f3a,f1b,f2b,f3b,frac,pq,ph;
   
    gam = 0.1971 - 0.0843*rn + 0.8460*rncur;
    a   = 1.5222 + 2.8553*rn + 2.3706*rn*rn + 0.9903*rn*rn*rn + 
            0.2250*rn*rn*rn*rn - 0.6038*rncur + 0.1749*om_v*(1.+w_lam);
    a = pow(10, a);
    b = pow(10, (-0.5642 + 0.5864*rn + 0.5716*rn*rn - 1.5474*rncur + 0.2279*om_v*(1.+w_lam)));
    c = pow(10, (0.3698 + 2.0404*rn + 0.8161*rn*rn + 0.5869*rncur));
    xmu = 0.;
    xnu = pow(10, (5.2105 + 3.6902*rn));
    alpha = abs(6.0835 + 1.3373*rn - 0.1959*rn*rn - 5.5274*rncur);  
    beta = 2.0379 - 0.7354*rn + 0.3157*rn*rn + 1.2490*rn*rn*rn + 0.3980*rn*rn*rn*rn - 0.1682*rncur;

    if (fabs(1-om_m) > 0.01) { 
        f1a  = pow(om_m, -0.0732);
        f2a  = pow(om_m, -0.1423);
        f3a  = pow(om_m, 0.0725);
        f1b  = pow(om_m, -0.0307);
        f2b  = pow(om_m, -0.0585);
        f3b  = pow(om_m, 0.0743);       
        frac = om_v/(1.-om_m);
        f1   = frac*f1b + (1-frac)*f1a;
        f2   = frac*f2b + (1-frac)*f2a;
        f3   = frac*f3b + (1-frac)*f3a;
    } else {
        f1 = 1.;
        f2 = 1.;
        f3 = 1.;
    }
    
    y = rk/rknl;
    ysqr = y*y;
    ph = a*pow(y, f1*3)/(1 + b*pow(y,f2) + pow(f3*c*y, 3-gam));
    ph = ph/(1 + xmu/y + xnu/ysqr);
    pq = plin*pow(1 + plin, beta)/(1 + plin*alpha)*exp(-y/4.0 - ysqr/8.0);
    *pnl = pq + ph;

    assert(finite(*pnl));
}
/*----------------------------------------------------------------------------*/
/* slope in the highly nonlinear regime, c.f. Smith et al (2002) eq. (61) */
double slope_NL(double rn, double rncur, double om_m, double om_v)
{
   double gam, f1a, f1b, frac, f1;

   gam = 0.86485 + 0.2989*rn + 0.1631*rncur;
   if(fabs(1-om_m)>0.01) {
      f1a = pow(om_m,(-0.0732));
      f1b = pow(om_m,(-0.0307));
      frac = om_v/(1.-om_m);  
      f1 = frac*f1b + (1-frac)*f1a;
   } else {
      f1 = 1.0;
   }

   return 3.0*(f1-1.0) + gam - 3.0;
}
/* ============================================================ *
 * Global variables needed in integrand functions for wint	
 * ============================================================ */

double rglob;

/* ============================================================ *
 * Calculates k_NL, n_eff, n_cur				
 * ============================================================ */ 
double int_for_wint2_knl(double logk)
{
    double krsqr, k;

    k = exp(logk);
    krsqr = (k*rglob)*(k*rglob);
    return Delta_L_onek(k)*exp(-krsqr);
}

/*----------------------------------------------------------------------------*/
double int_for_wint2_neff(double logk)
{   
    double krsqr, k;

    k = exp(logk);
    krsqr = (k*rglob)*(k*rglob);
    return Delta_L_onek(k)*2.0*krsqr*exp(-krsqr);
}

/*----------------------------------------------------------------------------*/
double int_for_wint2_ncur(double logk)
{
    double krsqr, k;

    k = exp(logk);
    krsqr = (k*rglob)*(k*rglob);
    return Delta_L_onek(k)*4.0*krsqr*(1.0-krsqr)*exp(-krsqr);
}

/*----------------------------------------------------------------------------*/
void wint2(double r, double *sig, double *d1, double *d2, double amp, int onlysig)
{
    const double kmin = 1.e-3;
    double kmax, logkmin, logkmax, s1, s2, s3;

    /* choose upper integration limit to where filter function dropped
     substantially */
    kmax  = sqrt(5.*log(10.))/r;
    if (kmax<8000.0) kmax = 8000.0;

    logkmin = log(kmin);
    logkmax = log(kmax);
    rglob = r;

    if (onlysig==1) {
       s1   = qromb1(int_for_wint2_knl, logkmin, logkmax);
       *sig = amp*sqrt(s1);
    } else s1 = 1./(amp*amp);   /* sigma = 1 */

    if (onlysig==0) {
       s2  = qromb1(int_for_wint2_neff, logkmin, logkmax);
       s3  = qromb1(int_for_wint2_ncur, logkmin, logkmax);
       *d1 = -s2/s1;
       *d2 = -(*d1)*(*d1) - s3/s1;
    }
 }
/*----------------------------------------------------------------------------*/
double dlog(double x)
{
   return log(x)/log(10.0);
}

/*----------------------------------------------------------------------------*/
/* Numerical Recipes Functions */
/*----------------------------------------------------------------------------*/
#define EPS 1.0e-6
#define JMAX 40
#define JMAXP (JMAX+1)
#define K 5

double qromb1(double (*func)(double), double a, double b)
{
	double ss,dss;
	double s[JMAXP],h[JMAXP+1];
	int j;

	h[1]=1.0;
	for (j=1;j<=JMAX;j++) {
		s[j]=trapzd1(func,a,b,j);
		if (j >= K) {
			polint(&h[j-K],&s[j-K],K,0.0,&ss,&dss);
			if (fabs(dss) <= EPS*fabs(ss)) return ss;
		}
		h[j+1]=0.25*h[j];
	}
	error("Too many steps in routine qromb");
	return 0.0;
}
#undef EPS
#undef JMAX
#undef JMAXP
#undef K

/*----------------------------------------------------------------------------*/
#define FUNC(x) ((*func)(x))

double trapzd1(double (*func)(double), double a, double b, int n)
{
	double x,tnm,sum,del;
	static double s;
	int it,j;

	if (n == 1) {
		return (s=0.5*(b-a)*(FUNC(a)+FUNC(b)));
	} else {
		for (it=1,j=1;j<n-1;j++) it <<= 1;
		tnm=it;
		del=(b-a)/tnm;
		x=a+0.5*del;
		for (sum=0.0,j=1;j<=it;j++,x+=del) {
		   sum += FUNC(x);
		}
		s=0.5*(s+(b-a)*sum/tnm);
		return s;
	}
}
#undef FUNC

/*----------------------------------------------------------------------------*/
void polint(double xa[], double ya[], int n, double x, double *y, double *dy)
{
	int i,m,ns=1;
	double den,dif,dift,ho,hp,w;
	double *c,*d;

	dif=fabs(x-xa[1]);
	c=vector(1,n);
	d=vector(1,n);
	for (i=1;i<=n;i++) {
		if ( (dift=fabs(x-xa[i])) < dif) {
			ns=i;
			dif=dift;
		}
		c[i]=ya[i];
		d[i]=ya[i];
	}
	*y=ya[ns--];
	for (m=1;m<n;m++) {
		for (i=1;i<=n-m;i++) {
			ho=xa[i]-x;
			hp=xa[i+m]-x;
			w=c[i+1]-d[i];
			if ( (den=ho-hp) == 0.0)
			  error("Error in routine polint");
			den=w/den;
			d[i]=hp*den;
			c[i]=ho*den;
		}
		*y += (*dy=(2*ns < (n-m) ? c[ns+1] : d[ns--]));
	}
	free_vector(d,1,n);
	free_vector(c,1,n);
}

/*----------------------------------------------------------------------------*/
void error(const char *s)
{
   fprintf(stderr, "error: ");
   fprintf(stderr, "%s", s);
   fprintf(stderr,"\n");
   exit(1);
}

/*----------------------------------------------------------------------------*/
double *vector(long nl, long nh)
/* allocate a double vector with subscript range v[nl..nh] */
{
	double *v;

	v=(double *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(double)));
	if (!v) error("allocation failure in vector()");
	return v-nl+NR_END;
}

/*----------------------------------------------------------------------------*/
void free_vector(double *v, long nl, long nh)
/* free a double vector allocated with vector() */
{
	free((FREE_ARG) (v+nl-NR_END));
}

/*----------------------------------------------------------------------------*/
double **matrix(long nrl, long nrh, long ncl, long nch)
/* allocate a double matrix with subscript range m[nrl..nrh][ncl..nch] */
{
	long i, nrow=nrh-nrl+1,ncol=nch-ncl+1;
	double **m;

	/* allocate pointers to rows */
	m=(double **) malloc((size_t)((nrow+NR_END)*sizeof(double*)));
	if (!m) error("allocation failure 1 in matrix()");
	m += NR_END;
	m -= nrl;

	/* allocate rows and set pointers to them */
	m[nrl]=(double *) malloc((size_t)((nrow*ncol+NR_END)*sizeof(double)));
	if (!m[nrl]) error("allocation failure 2 in matrix()");
	m[nrl] += NR_END;
	m[nrl] -= ncl;

	for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;

	/* return pointer to array of pointers to rows */
	return m;
}

/*----------------------------------------------------------------------------*/
void free_matrix(double **m, long nrl, long nrh, long ncl, long nch)
/* free a double matrix allocated by matrix() */
{
	free((FREE_ARG) (m[nrl]+ncl-NR_END));
	free((FREE_ARG) (m+nrl-NR_END));
}

/*----------------------------------------------------------------------------*/
void spline(double x[], double y[], int n, double yp1, double ypn, double y2[])
{
	int i,k;
	double p,qn,sig,un,*u;

	u=vector(1,n-1);
	if (yp1 > 0.99e30)
		y2[1]=u[1]=0.0;
	else {
		y2[1] = -0.5;
		u[1]=(3.0/(x[2]-x[1]))*((y[2]-y[1])/(x[2]-x[1])-yp1);
	}
	for (i=2;i<=n-1;i++) {
		sig=(x[i]-x[i-1])/(x[i+1]-x[i-1]);
		p=sig*y2[i-1]+2.0;
		y2[i]=(sig-1.0)/p;
		u[i]=(y[i+1]-y[i])/(x[i+1]-x[i]) - (y[i]-y[i-1])/(x[i]-x[i-1]);
		u[i]=(6.0*u[i]/(x[i+1]-x[i-1])-sig*u[i-1])/p;
	}
	if (ypn > 0.99e30)
		qn=un=0.0;
	else {
		qn=0.5;
		un=(3.0/(x[n]-x[n-1]))*(ypn-(y[n]-y[n-1])/(x[n]-x[n-1]));
	}
	y2[n]=(un-qn*u[n-1])/(qn*y2[n-1]+1.0);
	for (k=n-1;k>=1;k--)
		y2[k]=y2[k]*y2[k+1]+u[k];
	free_vector(u,1,n-1);
}

/*----------------------------------------------------------------------------*/
void splint(double xa[], double ya[], double y2a[], int n, double x, double *y)
{
	int klo,khi,k;
	double h,b,a;

	klo=1;
	khi=n;
	while (khi-klo > 1) {
		k=(khi+klo) >> 1;
		if (xa[k] > x) khi=k;
		else klo=k;
	}
	h=xa[khi]-xa[klo];
	if (h == 0.0) {
	   error("Bad xa input to routine splint");
	}
	a=(xa[khi]-x)/h;
	b=(x-xa[klo])/h;
	*y=a*ya[klo]+b*ya[khi]+
	  ((a*a*a-a)*y2a[klo]+(b*b*b-b)*y2a[khi])*(h*h)/6.0;
}
/*----------------------------------------------------------------------------*/

/* ============================================================ *
 * Interpolates f at the value x, where f is a double[n] array,	*
 * representing a function between a and b, stepwidth dx.	*
 * 'lower' and 'upper' are powers of a logarithmic power law	*
 * extrapolation. If no	extrapolation desired, set these to 0	*
 * ============================================================ */
double interpol(double *f, int n, double a, double b, double dx, double x,
	            double lower, double upper)
{
   double r;
   int  i;
   if (x < a) {
      if (lower==0.) {
	 error("value too small in interpol");
	 return 0.0;
      }
      return f[0] + lower*(x - a);
   }
   r = (x - a)/dx;
   i = (int)(floor(r));
   if (i+1 >= n) {
      if (upper==0.0) {
	 if (i+1==n) {
	    return f[i];  /* constant extrapolation */
	 } else {
	    error("value too big in interpol");
	    return 0.0;
	 }
      } else {
	 return f[n-1] + upper*(x-b); /* linear extrapolation */
      }
   } else {
      return (r - i)*(f[i+1] - f[i]) + f[i]; /* interpolation */
   }
}

/*----------------------------------------------------------------------------*/

/* ============================================================ *
 * like interpol, but f beeing a 2d-function			*
 * 'lower' and 'upper' are the powers of a power law extra-	*
 * polation in the first argument				*
 * ============================================================ */
double interpol2d(double **f,
		int nx, double ax, double bx, double dx, double x,
		int ny, double ay, double by, double dy, double y,
		double lower, double upper)
{
   double t, dt, s, ds;
   int i, j;
   if (x < ax) {
      error("value too small in interpol2d");
   }
   if (x > bx) {
      error("value too big in interpol2d");
   }
   t = (x - ax)/dx;
   i = (int)(floor(t));
   if (i+1 > nx || i < 0) error("index out of range in interpol");
   dt = t - i;
   if (y < ay) {
      return ((1.-dt)*f[i][0] + dt*f[i+1][0]) + (y-ay)*lower;
   } else if (y > by) {
      return ((1.-dt)*f[i][ny-1] + dt*f[i+1][ny-1]) + (y-by)*upper;
   }
   s = (y - ay)/dy;
   j = (int)(floor(s));
   ds = s - j;
   return (1.-dt)*(1.-ds)*f[i][j] +
     (1.-dt)*ds*f[i][j+1] +
     dt*(1.-ds)*f[i+1][j] +
     dt*ds*f[i+1][j+1];
}
/*----------------------------------------------------------------------------*/