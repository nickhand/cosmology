/*
 *  core.c
 *  
 *
 *  Created by Earl Lawrence on 9/17/09.
 *
 *  This program was prepared by Los Alamos National Security, LLC at Los Alamos National Laboratory (LANL) 
 *  under contract No. DE-AC52-06NA25396 with the U.S. Department of Energy (DOE). All rights in the program 
 *  are reserved by the DOE and Los Alamos National Security, LLC.  Permission is granted to the public to 
 *  copy and use this software without charge, provided that this Notice and any statement of authorship are 
 *  reproduced on all copies.  Neither the U.S. Government nor LANS makes any warranty, express or implied, 
 *  or assumes any liability or responsibility for the use of this software.
 *
 */
 
 
#include "stdio.h"
#include "math.h"
#include "stdlib.h"
#include "string.h"

void emulator_wrapper(double ombh2, double ommh2, double ns, double H0, double w, 
                    double sigma8, double z, int use_cmbh, double *ystar) {
    
    int type=2;
    double xstar[7], stuff[4], xstarcmb[6];
    int cmbh=use_cmbh;

    // the cosmo params
    xstar[0] = ombh2;
    xstar[1] = ommh2;
    xstar[2] = ns;
    if (cmbh == 0) xstar[3] = H0;
    xstar[4] = w;
    xstar[5] = sigma8;
    xstar[6] = z;
    
    if(cmbh == 1) {
        xstarcmb[0] = xstar[0];
        xstarcmb[1] = xstar[1];
        xstarcmb[2] = xstar[2];
        xstarcmb[3] = xstar[4];
        xstarcmb[4] = xstar[5];
        xstarcmb[5] = xstar[6];
        emu_noh(xstarcmb, ystar, &type);
        getH0fromCMB(xstarcmb, stuff);
        xstar[3] = 100.*stuff[3];
    } else {
        emu(xstar, ystar, &type);
    }
}
