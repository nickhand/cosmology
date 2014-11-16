Thank you for using our HOD P(k) emulator. Please see our paper on the
arxiv for further details. If you do use our emulator, please cite our
paper. Please send questions to jkwan@anl.gov.

To begin, you will need a version of GSL installed, which you can
download from here: http://www.gnu.org/software/gsl/. I have version
1.15, but any version that includes the gsl_matrix and gsl_spline
functions will do.

To make the executable, type: 
gcc -o emu.out -I/users/astro/jkwan/libs/gsl/include -L/users/astro/jkwan/libs/gsl/lib -lgsl -lgslcblas -lm -g emu.c main.c

The entire emulation process is contained in emu.c and main.c is just
a driver that opens the parameter file, checks that the HOD parameters
entered by the user is within the range of the code and writes the
output file. If you want to run a batch job, just swap out main.c with
your own driver, but be sure to include the same header files. Or you
can just incorporate the function emu(double *cosmoparams, double
outputredshift, double *output) into your code, but you need to
declare an array of 2025 doubles to hold the output. Note that emu
outputs in terms of Delta(k) = log10(k^1.5 P(k)/(4 pi^2)). This is
converted to P(k) in main.c, but if you only use emu.c, please keep
this in mind.

The k bins are defined in logk.h. The code assumes that the ordering
of parameters in *cosmoparams is the same as in the params.ini file.

To run the emulator, type: 
emu.out params.ini output.txt

params.ini contains the HOD parameters, in the following
order:

log10(Mcut)  
log10(M1)
sigma
kappa
alpha
z


Please write only one parameter per line. You can comment out a line
by prefacing it with the '#' character. The resultant 2-pt function
will be contained in output.txt. There will also be a short header
reminding you of your input parameters. 


The parameter ranges are:
12.85 < Mcut < 13.85  Msun
13.3 < M1 < 14.3      Msun
0.5 < sigma < 1.2
0.5 < kappa < 1.5
0.5 < alpha < 1.5
0 < z < 1

If you get it wrong, the code will helpfully remind you of the
parameter ranges again.

