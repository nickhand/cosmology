#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "logk.h"
#include "params.h"
#include <math.h>

int main(int argc, char**argv)
{

  double newparams[nparams]; 
  char inputs[256]; 
  char paramnames[5][20]; 
  double outputredshift; 

  if (argc < 3)
    {
      fprintf(stderr, "Some input files are missing.\n The correct arguments are ./emu.out params.ini output.txt\n"); 
      exit(1); 
    }

  sprintf(paramnames[0], "M_cut"); 
  sprintf(paramnames[1], "M1"); 
  sprintf(paramnames[2], "sigma"); 
  sprintf(paramnames[3], "kappa"); 
  sprintf(paramnames[4], "alpha"); 


  //read .ini file
  sprintf(inputs, "%s", argv[1]); 
  FILE *fpinputs = fopen(inputs,"r");
  if (fpinputs==NULL)
    {
      fprintf(stderr, "I can't find this parameter file: %s.\nExiting now. \n", argv[1]); 
      exit(1);
    }
  float buf; 
  char line[1000];
  int n=0;
  int c; 
  while (fgets(line, sizeof(line), fpinputs) || n < 6) {
    if (*line == '#') continue; // ignore comment line 
    if (sscanf(line, "%f", &buf) != 1) 
      {
	// one parameter per line only! 
	if (strlen(line)!=1) // ignore blank lines
	  {
	    fprintf(stderr, "I don't know how to read your file format.\nPlease modify it to contain one parameter per line only.\nFor comments, please use a '#' symbol at the start of the line.\n"); 
	    exit(1); 
	  }

      }
    else 
      {
	if (n==5)
	  outputredshift = buf; 
	else 
	  newparams[n] = buf; 
	n++; 
      }
  }

  if (n < 6)
    {
      fprintf(stderr, "I didn't read enough input parameters. \nSome lines may be missing from your input file.\n");  
      exit(1);
    }

  fclose(fpinputs);

  // check if parameters are within emulation range: 
  for (n = 0; n < 5; n++)
    {
      if (newparams[n] < min_design[n] || newparams[n] > max_design[n])
  	{
	  /* if (n == 3) */
	  /*   fprintf(stderr, "%s = %lf is outside of the emulation range: %f -- %f. \nPlease adjust your parameters accordingly.\n", paramnames[n], newparams[n], min_design[n], max_design[n]); */
	  /* else */
	    fprintf(stderr, "%s = %lf is outside of the emulation range: %f -- %f. \nPlease adjust your parameters accordingly.\n", paramnames[n], newparams[n], min_design[n], max_design[n]);
  	  fflush;
  	  exit(1);
  	}
      //      fprintf(stderr, "%s = %lf\n", paramnames[n], newparams[n]); 
    }
  if (outputredshift > 1 || outputredshift < 0)
    {
      fprintf(stderr, "%s = %f is outside of the emulation range: %f -- %f. \nPlease adjust your parameters accordingly.\n", paramnames[n], newparams[n], min_design[n], max_design[n]);
      fflush;
      exit(1);
    }
  

  char outputfile[256]; 
  sprintf(outputfile, "%s", argv[2]); 
  FILE *fp = fopen(outputfile,"w");
  if (fp==NULL)
    {
      fprintf(stderr, "I can't open this file for writing: %s.\nExiting now. \n", argv[2]); 
      exit(1);
    }
  else
      fprintf(fp, "# P_gal(k) at z = %.2f for: \n", outputredshift); 

  double *output_pk = malloc(nk*sizeof(double)); // this must be at least of length nk elements
  double h; 

  //write parameters to output file as a record of used parameters
  for (n = 0; n < 5; n++)
    fprintf(fp, "# %s = %f\n", paramnames[n], newparams[n]);

  // Now do the emulation! 
  emu(newparams, outputredshift, output_pk); 

  //  Convert back to P(k) from \Delta = k^1.5*P(k)/(4pi^2)

  double *k_unlogged = malloc(nk*sizeof(double)); 

  for (n = 0; n < nk; n++)
    {
      k_unlogged[n] = pow(10.,logk[n]);
      output_pk[n] = pow(10.,output_pk[n])/pow(k_unlogged[n], 1.5)*4*M_PI*M_PI;	  
      fprintf(fp,"%f %f\n", k_unlogged[n], output_pk[n]);  
    }

  fclose(fp);


  free(k_unlogged); 
  free(output_pk); 
  return(0);
}



