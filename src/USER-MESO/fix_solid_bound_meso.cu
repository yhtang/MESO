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
#include "fix_solid_bound_meso.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

MesoFixSolidBound::MesoFixSolidBound( LAMMPS *lmp, int narg, char **arg ):
    Fix( lmp, narg, arg ),
    MesoPointers( lmp )
{
    if( narg < 4 ) error->all( FLERR, "Illegal fix MesoFixSolidBound command" );

    x = y = z = false;
    force_kernel = UNSPECIFIED;
    for( int i = 0 ; i < narg ; i++ ) {
        if( !strcmp( arg[i], "x" ) ) x = true;
        else if( !strcmp( arg[i], "y" ) ) y = true;
        else if( !strcmp( arg[i], "z" ) ) z = true;
        else if( !strcmp( arg[i], "rho5rc1s1" ) ) force_kernel = RHO5_RC1_S1;
    }

    if( x == false && y == false && z == false ) {
        error->all( FLERR, "Incomplete fix wall command: dimension unspecified" );
    }
    if( force_kernel == UNSPECIFIED ) {
        error->all( FLERR, "Incomplete fix wall command: force kernel unspecified" );
    }

    nevery = 1;
}

int MesoFixSolidBound::setmask()
{
    int mask = 0;
    mask |= FixConst::POST_FORCE;
    mask |= FixConst::PRE_EXCHANGE;
    mask |= FixConst::END_OF_STEP;
    return mask;
}

void MesoFixSolidBound::init()
{
    if( strcmp( update->integrate_style, "respa" ) == 0 ) {
        fprintf( stderr, "<MESO> RESPA not supported in MesoFixSolidBound. %s %d\n", __FILE__, __LINE__ );
    }
}

void MesoFixSolidBound::setup( int vflag )
{
    if( strcmp( update->integrate_style, "respa" ) == 0 ) {
        fprintf( stderr, "<MESO> RESPA not supported in MesoFixSolidBound. %s %d\n", __FILE__, __LINE__ );
    }
    post_force( vflag );
}

template<class FORCE_KERNEL>
__global__ void gpu_fix_solid_wall_force(
    r64* __restrict coord_x,
    r64* __restrict coord_y,
    r64* __restrict coord_z,
    r64* __restrict force_x,
    r64* __restrict force_y,
    r64* __restrict force_z,
    int* __restrict mask,
    const double3 boxhi,
    const double3 boxlo,
    const bool x,
    const bool y,
    const bool z,
    const int groupbit,
    const int n_all )
{
    FORCE_KERNEL force_kernel;
    double d = force_kernel.d;
    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_all; i += gridDim.x * blockDim.x ) {
        if( mask[i] & groupbit ) {
            if( x ) {
                double h;
                h = coord_x[i] - boxlo.x;
                if( h <= d ) force_x[i] += force_kernel( h );
                h = boxhi.x - coord_x[i];
                if( h <= d ) force_x[i] -= force_kernel( h );
            }
            if( y ) {
                double h;
                h = coord_y[i] - boxlo.y;
                if( h <= d ) force_y[i] += force_kernel( h );
                h = boxhi.y - coord_y[i];
                if( h <= d ) force_y[i] -= force_kernel( h );
            }
            if( z ) {
                double h;
                h = coord_z[i] - boxlo.z;
                if( h <= d ) force_z[i] += force_kernel( h );
                h = boxhi.z - coord_z[i];
                if( h <= d ) force_z[i] -= force_kernel( h );
            }
        }
    }
}

void MesoFixSolidBound::boundary_force()
{
    double3 hi = make_double3( domain->boxhi[0], domain->boxhi[1], domain->boxhi[2] );
    double3 lo = make_double3( domain->boxlo[0], domain->boxlo[1], domain->boxlo[2] );

    if( force_kernel == RHO5_RC1_S1 ) {
        static GridConfig grid_cfg;
        if( !grid_cfg.x ) {
            grid_cfg = meso_device->configure_kernel( gpu_fix_solid_wall_force<Rho5rc1s1> );
        }

        gpu_fix_solid_wall_force<Rho5rc1s1> <<< grid_cfg.x, grid_cfg.y, 0, meso_device->stream() >>> (
            meso_atom->dev_coord(0),
            meso_atom->dev_coord(1),
            meso_atom->dev_coord(2),
            meso_atom->dev_force(0),
            meso_atom->dev_force(1),
            meso_atom->dev_force(2),
            meso_atom->dev_mask,
            hi, lo,
            x, y, z,
            groupbit,
            atom->nlocal );
    }
}

// bounce-forward
__global__ void gpu_fix_solid_wall_bounce(
    int *Tag,
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
    const int nAll )
{
    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < nAll; i += gridDim.x * blockDim.x ) {
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
                if( coord_z[i] <= boxlo.z ) {
                    veloc_z[i] = fabs( veloc_z[i] );
                    coord_z[i] = 2. * boxlo.z - coord_z[i];
                } else if( coord_z[i] >= boxhi.z ) {
                    veloc_z[i] = -fabs( veloc_z[i] );
                    coord_z[i] = 2. * boxhi.z - coord_z[i];
                }
            }
        }
    }
}

void MesoFixSolidBound::bounce_back()
{
    static GridConfig grid_cfg;
    if( !grid_cfg.x ) {
        grid_cfg = meso_device->occu_calc.right_peak( 0, gpu_fix_solid_wall_bounce, 0, cudaFuncCachePreferL1 );
        cudaFuncSetCacheConfig( gpu_fix_solid_wall_bounce, cudaFuncCachePreferL1 );
    }

    double3 hi = make_double3( domain->boxhi[0], domain->boxhi[1], domain->boxhi[2] );
    double3 lo = make_double3( domain->boxlo[0], domain->boxlo[1], domain->boxlo[2] );

    gpu_fix_solid_wall_bounce <<< grid_cfg.x, grid_cfg.y, 0, meso_device->stream() >>> (
        meso_atom->dev_tag,
        meso_atom->dev_coord(0), meso_atom->dev_coord(1), meso_atom->dev_coord(2),
        meso_atom->dev_veloc(0), meso_atom->dev_veloc(1), meso_atom->dev_veloc(2),
        meso_atom->dev_mask,
        hi, lo,
        x, y, z,
        groupbit,
        atom->nlocal );
}

void MesoFixSolidBound::post_integrate()
{
    bounce_back();
}

void MesoFixSolidBound::end_of_step()
{
    bounce_back();
}

void MesoFixSolidBound::pre_exchange()
{
    bounce_back();
}

void MesoFixSolidBound::post_force( int vflag )
{
    boundary_force();
}

