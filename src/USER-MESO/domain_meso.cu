#include "mpi.h"
#include "stdlib.h"
#include "string.h"
#include "stdio.h"
#include "math.h"
#include "style_region.h"
#include "atom.h"
#include "force.h"
#include "kspace.h"
#include "update.h"
#include "modify.h"
#include "fix.h"
#include "fix_deform.h"
#include "region.h"
#include "lattice.h"
#include "comm.h"
#include "universe.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"

#include "domain_meso.h"

using namespace LAMMPS_NS;

MesoDomain::MesoDomain( class LAMMPS *lmp ) : Domain( lmp ), MesoPointers( lmp )
{
}

void MesoDomain::pbc()
{
    double *lo, *hi, *period;
    int nlocal = atom->nlocal;
    double **x = atom->x;
    double **v = atom->v;
    int *mask = atom->mask;
    tagint *image = atom->image;

    if( triclinic == 0 ) {
        lo = boxlo;
        hi = boxhi;
        period = prd;
    } else {
        lo = boxlo_lamda;
        hi = boxhi_lamda;
        period = prd_lamda;
    }

    static ThreadTuner &o = meso_device->tuner( "MesoDomain::pbc" );
    size_t ntd = o.bet();
    double t1 = meso_device->get_time_omp();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        if( tid < ntd ) {
            int2 work = split_work( 0, nlocal, tid, ntd );
            for( int i = work.x; i < work.y; i++ ) {
                tagint idim, otherdims;
                if( xperiodic ) {
                    if( x[i][0] < lo[0] ) {
                        x[i][0] += period[0];
                        if( deform_vremap && mask[i] & deform_groupbit ) v[i][0] += h_rate[0];
                        idim = image[i] & IMGMASK;
                        otherdims = image[i] ^ idim;
                        idim--;
                        idim &= IMGMASK;
                        image[i] = otherdims | idim;
                    }
                    if( x[i][0] >= hi[0] ) {
                        x[i][0] -= period[0];
                        x[i][0] = MAX( x[i][0], lo[0] );
                        if( deform_vremap && mask[i] & deform_groupbit ) v[i][0] -= h_rate[0];
                        idim = image[i] & IMGMASK;
                        otherdims = image[i] ^ idim;
                        idim++;
                        idim &= IMGMASK;
                        image[i] = otherdims | idim;
                    }
                }

                if( yperiodic ) {
                    if( x[i][1] < lo[1] ) {
                        x[i][1] += period[1];
                        if( deform_vremap && mask[i] & deform_groupbit ) {
                            v[i][0] += h_rate[5];
                            v[i][1] += h_rate[1];
                        }
                        idim = ( image[i] >> IMGBITS ) & IMGMASK;
                        otherdims = image[i] ^ ( idim << IMGBITS );
                        idim--;
                        idim &= IMGMASK;
                        image[i] = otherdims | ( idim << IMGBITS );
                    }
                    if( x[i][1] >= hi[1] ) {
                        x[i][1] -= period[1];
                        x[i][1] = MAX( x[i][1], lo[1] );
                        if( deform_vremap && mask[i] & deform_groupbit ) {
                            v[i][0] -= h_rate[5];
                            v[i][1] -= h_rate[1];
                        }
                        idim = ( image[i] >> IMGBITS ) & IMGMASK;
                        otherdims = image[i] ^ ( idim << IMGBITS );
                        idim++;
                        idim &= IMGMASK;
                        image[i] = otherdims | ( idim << IMGBITS );
                    }
                }

                if( zperiodic ) {
                    if( x[i][2] < lo[2] ) {
                        x[i][2] += period[2];
                        if( deform_vremap && mask[i] & deform_groupbit ) {
                            v[i][0] += h_rate[4];
                            v[i][1] += h_rate[3];
                            v[i][2] += h_rate[2];
                        }
                        idim = image[i] >> IMG2BITS;
                        otherdims = image[i] ^ ( idim << IMG2BITS );
                        idim--;
                        idim &= IMGMASK;
                        image[i] = otherdims | ( idim << IMG2BITS );
                    }
                    if( x[i][2] >= hi[2] ) {
                        x[i][2] -= period[2];
                        x[i][2] = MAX( x[i][2], lo[2] );
                        if( deform_vremap && mask[i] & deform_groupbit ) {
                            v[i][0] -= h_rate[4];
                            v[i][1] -= h_rate[3];
                            v[i][2] -= h_rate[2];
                        }
                        idim = image[i] >> IMG2BITS;
                        otherdims = image[i] ^ ( idim << IMG2BITS );
                        idim++;
                        idim &= IMGMASK;
                        image[i] = otherdims | ( idim << IMG2BITS );
                    }
                }
            }
        }
    }

    double t2 = meso_device->get_time_omp();
    if( meso_device->warmed_up() ) o.learn( ntd, t2 - t1 );
}
