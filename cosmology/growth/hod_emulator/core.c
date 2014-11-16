#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "logk.h"
#include "params.h"
#include <math.h>

void hod_emulator_wrapper(double M_cut, double M1, double sigma, double kappa, 
                            double alpha, double outputredshift, double *output_pk)
{

    double newparams[nparams];    
    int n = 0;
    int nk = 2025;
  
    // set the parameters
    newparams[0] = M_cut;
    newparams[1] = M1;
    newparams[2] = sigma;
    newparams[3] = kappa;
    newparams[4] = alpha;


    // check if parameters are within emulation range: 
    for (n = 0; n < 5; n++) {
        if (newparams[n] < min_design[n] || newparams[n] > max_design[n]) {

            // fprintf(stderr, "%s = %lf is outside of the emulation range: %f -- %f. \nPlease adjust your parameters accordingly.\n", paramnames[n], newparams[n], min_design[n], max_design[n]);
            //          fflush;
    	    exit(1);
    	}
    }
    if (outputredshift > 1 || outputredshift < 0) {
      // fprintf(stderr, "%s = %f is outside of the emulation range: %f -- %f. \nPlease adjust your parameters accordingly.\n", paramnames[n], newparams[n], min_design[n], max_design[n]);
      // fflush;
      exit(1);
    }

    // Now do the emulation! 
    hod_emu(newparams, outputredshift, output_pk); 
    
    //  Convert back to P(k) from \Delta = k^1.5*P(k)/(4pi^2)
    for (n = 0; n < nk; n++) {
        
        double delta = output_pk[n];
        //printf("delta = %f\n", delta);
        //printf("logk = %f\n", logk[n]);
        output_pk[n] = pow(10., logk[n]);
        output_pk[n+nk] = pow(10., delta)/pow(output_pk[n], 1.5)*4*M_PI*M_PI;	   
    }
  
}



