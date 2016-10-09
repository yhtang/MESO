#include "mpi.h"
#include "stdio.h"
#include "string.h"
#include "force.h"
#include "update.h"
#include "error.h"
#include "neighbor.h"
#include "domain.h"

#include "atom_meso.h"
#include "comm_meso.h"
#include "atom_vec_meso.h"
#include "engine_meso.h"
#include "neighbor_meso.h"
#include "fix_bounce_back_meso.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

MesoFixBounceBack::MesoFixBounceBack( LAMMPS *lmp, int narg, char **arg ):
    Fix( lmp, narg, arg ),
    MesoPointers( lmp )
{
    if( narg < 4 ) error->all( FLERR, "Illegal fix MesoFixBounceBack command" );

    x = y = z = false;
    for( int i = 0 ; i < narg ; i++ ) {
        if( !strcmp( arg[i], "x" ) ) x = true;
        else if( !strcmp( arg[i], "y" ) ) y = true;
        else if( !strcmp( arg[i], "z" ) ) z = true;
    }

    if( x == false && y == false && z == false ) {
        error->all( FLERR, "Incomplete fix wall command: dimension unspecified" );
    }

    nevery = 1;
}

int MesoFixBounceBack::setmask()
{
    int mask = 0;
    mask |= FixConst::PRE_EXCHANGE;
    mask |= FixConst::END_OF_STEP;
    return mask;
}

void MesoFixBounceBack::init()
{
    if( strcmp( update->integrate_style, "respa" ) == 0 ) {
        fprintf( stderr, "<MESO> RESPA not supported in MesoFixBounceBack. %s %d\n", __FILE__, __LINE__ );
    }
}

void MesoFixBounceBack::setup( int vflag )
{
    if( strcmp( update->integrate_style, "respa" ) == 0 ) {
        fprintf( stderr, "<MESO> RESPA not supported in MesoFixBounceBack. %s %d\n", __FILE__, __LINE__ );
    }
    post_force( vflag );
}

// bounce-forward
__global__ void gpu_fix_solid_wall_bounce(
    r64* __restrict coord_x,
    r64* __restrict coord_y,
    r64* __restrict coord_z,
    r64* __restrict veloc_x,
    r64* __restrict veloc_y,
    r64* __restrict veloc_z,
    int* __restrict Mask,
    const double3 boxhi,
    const double3 boxlo,
    const bool x,
    const bool y,
    const bool z,
    const int groupbit,
    const int n_all )
{
    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_all; i += gridDim.x * blockDim.x ) {
        if( Mask[i] & groupbit ) {
            if( x ) {
                if( coord_x[i] <= boxlo.x ) {
                    veloc_x[i] = fabs( veloc_x[i] );
                    coord_x[i] = 2. * boxlo.x - coord_x[i];
                } else if( coord_x[i] >= boxhi.x ) {
                    veloc_x[i] = -fabs( veloc_x[i] );
                    coord_x[i] = 2. * boxhi.x - coord_x[i];
                }
            }
            if( y ) {
                if( coord_y[i] <= boxlo.y ) {
                    veloc_y[i] = fabs( veloc_y[i] );
                    coord_y[i] = 2. * boxlo.y - coord_y[i];
                } else if( coord_y[i] >= boxhi.y ) {
                    veloc_y[i] = -fabs( veloc_y[i] );
                    coord_y[i] = 2. * boxhi.y - coord_y[i];
                }
            }
            if( z ) {
            	//r64 z0 = coord_z[i];
                if( coord_z[i] <= boxlo.z ) {
                    veloc_z[i] = fabs( veloc_z[i] );
                    coord_z[i] = 2. * boxlo.z - coord_z[i];
                } else if( coord_z[i] >= boxhi.z ) {
                    veloc_z[i] = -fabs( veloc_z[i] );
                    coord_z[i] = 2. * boxhi.z - coord_z[i];
                }
                //if ( coord_z[i] <= boxlo.z || coord_z[i] >= boxhi.z ) {
                //	printf("particle %d still out of box after bouncing back, z0 = %lf\n", z0);
                //}
            }
        }
    }
}

void MesoFixBounceBack::bounce_back()
{
    static GridConfig grid_cfg;
    if( !grid_cfg.x ) {
        grid_cfg = meso_device->occu_calc.right_peak( 0, gpu_fix_solid_wall_bounce, 0, cudaFuncCachePreferL1 );
        cudaFuncSetCacheConfig( gpu_fix_solid_wall_bounce, cudaFuncCachePreferL1 );
    }

    double3 hi = make_double3( domain->boxhi[0], domain->boxhi[1], domain->boxhi[2] );
    double3 lo = make_double3( domain->boxlo[0], domain->boxlo[1], domain->boxlo[2] );

    gpu_fix_solid_wall_bounce <<< grid_cfg.x, grid_cfg.y, 0, meso_device->stream() >>> (
        meso_atom->dev_coord(0), meso_atom->dev_coord(1), meso_atom->dev_coord(2),
        meso_atom->dev_veloc(0), meso_atom->dev_veloc(1), meso_atom->dev_veloc(2),
        meso_atom->dev_mask,
        hi, lo,
        x, y, z,
        groupbit,
        atom->nlocal );
}

void MesoFixBounceBack::end_of_step()
{
    bounce_back();
}

void MesoFixBounceBack::pre_exchange()
{
    bounce_back();
}

