#include <stdlib.h>
#include <stdio.h>
#include "emu.h"
#include "params.h"
#include "design.h"
#include <math.h>

float min_design[5] = {12.85, 13.3, 0.5, 0.5, 0.5};
float max_design[5] = {13.85, 14.3, 1.2, 1.5, 1.5};
int nk = 2025;   // Number of mass bins
int numPC = 5;    // Number of pricipal components
int nmodels = 100; // How many models to cover parameter space
int nparams = 5; // Number of cosmological parameters

int make_sigma_w(float dparams[][nparams], double *newparams, gsl_matrix *V11, gsl_matrix *V21, int PCnow, int numz);
int read_design(FILE *fp, float design[][nparams], int norm);
int invert_matrix(int size, gsl_matrix *A, gsl_matrix *A_inv);

int hod_emu(double *newparams, double outputredshift, double *output_pk)
{
  int n; 
  char inputs[256]; 
  // float designparams[nmodels][nparams]; // this is nmodels x nparameters
  // 
  // //read in the design matrix
  // FILE *fpdesign = fopen("s-lhs.100.5_1", "r");
  // if (fpdesign == NULL)
  //    printf("Error in opening file s-lhs.100.5_1\n");
  // int norm = 0; // 1 = normalise the design, 0 = design is already ok
  // read_design(fpdesign, designparams, norm);
  // fclose(fpdesign); 

    // xstar contains the 5 emulator parameters plus the red shift.
  double xstar[6], stuff[4]; 
  xstar[0] = newparams[0];
  xstar[1] = newparams[1];
  xstar[2] = newparams[2];
  xstar[3] = newparams[4];
  xstar[4] = newparams[3];
  xstar[5] = outputredshift;

  //  normalise new inputs:
  for (n = 0; n < 5; n++)
    {
      newparams[n]-=min_design[n];
      newparams[n]/=(max_design[n]-min_design[n]);
    }

  gsl_matrix * V11 = gsl_matrix_alloc (nmodels, nmodels);
  gsl_matrix * V11_inv = gsl_matrix_alloc (nmodels, nmodels);
  gsl_matrix * V21 = gsl_matrix_alloc (1, nmodels); // leave this as a matrix in case several interpolations are wanted at once; 


  double wpred[numPC]; //predicted weights at new cM relation
  int numz = 0; // redshift 
  double *pkpred = calloc(nk*6,sizeof(double));  // predicted cM relation at z = 0, 1
  
  for (numz = 0; numz < 6; numz++)
    {
      for (n = 0; n < numPC; n++)
	{
	  wpred[n] = 0;

	  make_sigma_w(designparams, newparams, V11, V21,n, numz);

	  invert_matrix(nmodels, V11, V11_inv); 

	  int i, j;
	  double *dummyreslt  = calloc(nmodels,sizeof(double));
	  for (i = 0; i < nmodels; i++)
	    {
	      for(j = 0; j < nmodels; j++)
		dummyreslt[i] += gsl_matrix_get(V11_inv, i, j)*what[j+n*nmodels][numz];
	    }
	  for (i = 0; i < nmodels; i++)
	    wpred[n] += gsl_matrix_get(V21, 0, i)*dummyreslt[i];

	  free(dummyreslt);
	  /* fprintf(stderr, "%f\n", wpred[n]);  */
	}

      for (n = 0; n < nk; n++)
	{
	  int m; 
	  pkpred[6*n+numz] = ymean[n][numz];
	  for (m = 0; m < numPC; m++)
	    {
	      pkpred[6*n+numz] += phi[n][numz*numPC+m]*wpred[m]*ysimstd[numz]; 
	    }
	}

    }

  gsl_spline *spline = gsl_spline_alloc(gsl_interp_linear, 6); 
  gsl_interp_accel *acc  = gsl_interp_accel_alloc ();

  double scalefactor[6] = {0.4985, 0.6086, 0.6974, 0.8051, 0.9083, 1.0};
  double redshifts[6];

  for (n = 0; n < 6; n++)
    redshifts[n] = 1./scalefactor[n]-1; 

  double output_scalefactor = 1./(outputredshift+1.); 


  for (n = 0; n < nk; n++)
    {
      gsl_spline_init (spline, scalefactor, &(pkpred[6*n]), 6);
      output_pk[n] = gsl_spline_eval(spline, output_scalefactor, acc);
      /* fprintf(fp_test, "%f %f %f %f %f %f\n", pkpred[6*n+0], pkpred[6*n+1], pkpred[6*n+2], pkpred[6*n+3], pkpred[6*n+4], pkpred[6*n+5]);  */
      gsl_interp_accel_reset(acc);
    }

  // undo normalisation (to avoid confusion in driver program)
  for (n = 0; n < 5; n++)
    {
      newparams[n]*=(max_design[n]-min_design[n]);
      newparams[n]+=min_design[n];
      /* if(n==3) */
      /* 	newparams[n] = -newparams[n]; */
      //      fprintf(stderr, "%lf\n", newparams[n]); 
    }


  /* gsl_interp_spline_free(spline); */
  gsl_interp_accel_free(acc);
  gsl_matrix_free(V11);
  gsl_matrix_free(V11_inv);
  gsl_matrix_free(V21);
  free(pkpred);

  return(0);
}

int read_design(FILE *fp, float design[][nparams], int norm)
{

  // this is the design matrix
  // if norm==1 then the matrix needs to be normalized.
  int i, j, k; 
  for (i = 0; i <  nmodels; i++) {
      for (j = 0; j <  nparams;j++) {
	      //fscanf(fp, "%f", &(design[i][j])); 
	      printf("design[%d][%d] = %f\n", i, j, design[i][j]);
	      if (norm) {
	          design[i][j]-=min_design[j]; 
	          design[i][j]/=(max_design[j]-min_design[j]); 
	      }
      }
  }
  return(0); 
}

int make_sigma_w(float dparams[][nparams], double *newparams, gsl_matrix *V11, gsl_matrix *V21, int PCnow, int numz)
{
  // Do each PC individually. 

  int i, j, k; 
  for (i = 0; i < nmodels; i++)
    {
    for (j = 0; j < nmodels; j++)
    /* for (j = 0; j < i; j++) */
      {
	double V11dummy = 1.; 
	for (k = 0; k < nparams; k++)
	  {
	    double distance = dparams[i][k]-dparams[j][k]; // always substract the same type of parameter from the same type i.e. Omegam_1 - Omegam_2 not Omegam_1-w 
	    distance = 4.*distance*distance; 
	    V11dummy *= pow(rho_w[k][numz*numPC+PCnow], distance); 
	  }
	V11dummy /= lambda_w[numz*numPC+PCnow]; 
	if (i==j)
	  V11dummy += 1./lambdaP[numz];
	  //	V11dummy += invphi[i+PCnow*nmodels][j+PCnow*nmodels]/lambdaP; //check this part..
	gsl_matrix_set (V11, i, j, V11dummy);
      }
    }


  for (j = 0; j < nmodels; j++)
    {
      double V21dummy = 1.; 
      for (k = 0; k < nparams; k++)
	{
	  double distance = dparams[j][k]-newparams[k];
	  distance = 4.*distance*distance; 
	  V21dummy *= pow(rho_w[k][numz*numPC+PCnow], distance); 
	}
      V21dummy /= lambda_w[numz*numPC+PCnow]; 
      gsl_matrix_set(V21, 0, j, V21dummy); 
    }

  return(0); 
}

int invert_matrix(int size, gsl_matrix *A, gsl_matrix *A_inv)
{

  gsl_matrix *Adummy = gsl_matrix_alloc(size,size);
  gsl_matrix *Adummy2 = gsl_matrix_alloc(size,size);
  
  gsl_matrix_memcpy (Adummy, A); 
  /* gsl_linalg_cholesky_decomp (Adummy);  // Adummy will be overwritten  */
  /* gsl_linalg_cholesky_invert (Adummy);  // This is just the inverse of the lower half of the decomposed matrix */
  gsl_vector *work = gsl_vector_alloc(size);
  gsl_matrix *V = gsl_matrix_alloc(size,size); // V is untransposed
  gsl_vector *Sdiag = gsl_vector_alloc(size); 
  gsl_matrix *S = gsl_matrix_alloc(size,size); 

  gsl_linalg_SV_decomp(Adummy, V, Sdiag, work); //A_inv is actually U: A = USV', Adummy becomes U on output;
  int i,j;
  double reslt = 0;

  gsl_matrix_set_zero(S);
  for (i=0; i < size; i++)
    {
      double reslt = gsl_vector_get(Sdiag,i);
      gsl_matrix_set(S,i,i, 1./reslt);
    }

  gsl_blas_dgemm(CblasNoTrans, CblasTrans,1.0,S, Adummy, 0.0, Adummy2);
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,1.0,V, Adummy2, 0.0, A_inv);
	  
  return(0); 
}


