#include "mpi.h"
#include "stdio.h"
#include "string.h"
#include "force.h"
#include "update.h"
#include "error.h"
#include "domain.h"

#include "atom_meso.h"
#include "comm_meso.h"
#include "atom_vec_meso.h"
#include "engine_meso.h"
#include "fix_plane_meso.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

MesoFixPlane::MesoFixPlane( LAMMPS *lmp, int narg, char **arg ):
    Fix( lmp, narg, arg ),
    MesoPointers( lmp )
{
    if( narg < 6 ) error->all( FLERR, "Illegal fix MesoFixPlane command" );

    nx = ny = nz = 0.0;
    d = 0.0;
    f = 0.0;
    for( int i = 0 ; i < narg ; i++ ) {
        if( !strcmp( arg[i], "n" ) ) {
            if( i + 3 >= narg ) error->all( FLERR, "Incomplete fix plane command after 'n'" );
            nx = atof( arg[++i] );
            ny = atof( arg[++i] );
            nz = atof( arg[++i] );
        } else if( !strcmp( arg[i], "d" ) ) {
            if( ++i >= narg ) error->all( FLERR, "Incomplete fix plane command after 'd'" );
            d = atof( arg[i] );
        } else if( !strcmp( arg[i], "f" ) ) {
            if( ++i >= narg ) error->all( FLERR, "Incomplete fix plane command after 'f'" );
            f = atof( arg[i] );
        }
    }

    // f == 0.0 && d == 0.0 is allowed, that reduced to simple bounce-forward
    if( nx == 0.0 && ny == 0.0 && nz == 0.0 ) {
        error->all( FLERR, "Incomplete fix plane command: insufficient arguments" );
    } else {
        r64 norm = sqrt( nx * nx + ny * ny + nz * nz );
        nx /= norm;
        ny /= norm;
        nz /= norm;
    }

    nevery = 1;
}

int MesoFixPlane::setmask()
{
    int mask = 0;
    mask |= FixConst::POST_FORCE;
    mask |= FixConst::PRE_EXCHANGE;
    mask |= FixConst::END_OF_STEP;
    return mask;
}

void MesoFixPlane::init()
{
    if( strcmp( update->integrate_style, "respa" ) == 0 ) {
        fprintf( stderr, "<MESO> RESPA not supported in MesoFixPlane. %s %d\n", __FILE__, __LINE__ );
    }
}

void MesoFixPlane::setup( int vflag )
{
    if( strcmp( update->integrate_style, "respa" ) == 0 ) {
        fprintf( stderr, "<MESO> RESPA not supported in MesoFixPlane. %s %d\n", __FILE__, __LINE__ );
    }
    post_force( vflag );
}

#define SQRT3 1.732050808

__global__ void gpu_fix_plane_force(
    r64* __restrict coord_x,
    r64* __restrict coord_y,
    r64* __restrict coord_z,
    r64* __restrict force_x,
    r64* __restrict force_y,
    r64* __restrict force_z,
    int* __restrict mask,
    const r64 d,
    const r64 dinv,
    const r64 f,
    const r64 nx,
    const r64 ny,
    const r64 nz,
    const int groupbit,
    const int n_all )
{
    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_all; i += gridDim.x * blockDim.x ) {
        if( mask[i] & groupbit ) {
            r64 x = coord_x[i];
            r64 y = coord_y[i];
            r64 z = coord_z[i];
            r64 h = x * nx + y * ny + z * nz;
            if( h < d ) {
                r64 force = f * erfcf( ( h - 0.5 * d ) * dinv * SQRT3 );
                force_x[i] += force * nx;
                force_y[i] += force * ny;
                force_z[i] += force * nz;
            }
        }
    }
}

void MesoFixPlane::plane_force()
{
    static GridConfig grid_cfg;
    if( !grid_cfg.x ) {
        grid_cfg = meso_device->occu_calc.right_peak( 0, gpu_fix_plane_force, 0, cudaFuncCachePreferL1 );
        cudaFuncSetCacheConfig( gpu_fix_plane_force, cudaFuncCachePreferL1 );
    }

    if( !f ) return;

    gpu_fix_plane_force <<< grid_cfg.x, grid_cfg.y, 0, meso_device->stream() >>> (
        meso_atom->dev_coord(0), meso_atom->dev_coord(1), meso_atom->dev_coord(2),
        meso_atom->dev_force(0), meso_atom->dev_force(1), meso_atom->dev_force(2),
        meso_atom->dev_mask,
        d,
        1.0 / d,
        f,
        nx, ny, nz,
        groupbit,
        atom->nlocal );
}

// bounce-forward
__global__ void gpu_fix_plane_bounce(
    int *Tag,
    r64* __restrict coord_x,
    r64* __restrict coord_y,
    r64* __restrict coord_z,
    r64* __restrict veloc_x,
    r64* __restrict veloc_y,
    r64* __restrict veloc_z,
    int* __restrict mask,
    const r64 d,
    const r64 nx,
    const r64 ny,
    const r64 nz,
    const int groupbit,
    const int nAll )
{
    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < nAll; i += gridDim.x * blockDim.x ) {
        if( mask[i] & groupbit ) {
            r64 x = coord_x[i];
            r64 y = coord_y[i];
            r64 z = coord_z[i];
            r64 h = x * nx + y * ny + z * nz;
            if( h < d ) {
                r64 vx = veloc_x[i];
                r64 vy = veloc_y[i];
                r64 vz = veloc_z[i];
                r64 vmod = vx * nx + vy * ny + vz * nz;
                veloc_x[i] = 2.*vmod * nx - vx;
                veloc_y[i] = 2.*vmod * ny - vy;
                veloc_z[i] = 2.*vmod * nz - vz;
            }
        }
    }
}

void MesoFixPlane::bounce_back()
{
    static GridConfig grid_cfg;
    if( !grid_cfg.x ) {
        grid_cfg = meso_device->occu_calc.right_peak( 0, gpu_fix_plane_bounce, 0, cudaFuncCachePreferL1 );
        cudaFuncSetCacheConfig( gpu_fix_plane_bounce, cudaFuncCachePreferL1 );
    }

    double3 hi = make_double3( domain->boxhi[0], domain->boxhi[1], domain->boxhi[2] );
    double3 lo = make_double3( domain->boxlo[0], domain->boxlo[1], domain->boxlo[2] );

    gpu_fix_plane_bounce <<< grid_cfg.x, grid_cfg.y, 0, meso_device->stream() >>> (
        meso_atom->dev_tag,
        meso_atom->dev_coord(0), meso_atom->dev_coord(1), meso_atom->dev_coord(2),
        meso_atom->dev_veloc(0), meso_atom->dev_veloc(1), meso_atom->dev_veloc(2),
        meso_atom->dev_mask,
        d,
        nx, ny, nz,
        groupbit,
        atom->nlocal );
}

void MesoFixPlane::post_integrate()
{
    bounce_back();
}

void MesoFixPlane::end_of_step()
{
    bounce_back();
}

void MesoFixPlane::pre_exchange()
{
    bounce_back();
}

void MesoFixPlane::post_force( int vflag )
{
    plane_force();
}

