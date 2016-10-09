#include "mpi.h"
#include "stdio.h"
#include "string.h"
#include "force.h"
#include "update.h"
#include "respa.h"
#include "error.h"

#include "atom_meso.h"
#include "atom_vec_meso.h"
#include "comm_meso.h"
#include "engine_meso.h"
#include "fix_opt_meso.h"

using namespace LAMMPS_NS;

FixOptMeso::FixOptMeso( LAMMPS *lmp, int narg, char **arg ) :
    Fix( lmp, narg, arg ),
    MesoPointers( lmp ),
    dev_max_accl( lmp, "MesoFixOpt::dev_max_force" )
{
    dev_max_accl.grow( 1 );

    max_move    = 0.1;
    noise_level = 0.0;

    for( int i = 0 ; i < narg ; i++ ) {
        if( !strcmp( arg[i], "dr" ) && i + 1 < narg ) {
            max_move = atof( arg[++i] );
        }
        if( !strcmp( arg[i], "noise" ) && i + 1 < narg ) {
            noise_level = atof( arg[++i] );
        }
    }

    time_integrate = 1;
}

FixOptMeso::~FixOptMeso()
{
}

void FixOptMeso::init()
{
}

int FixOptMeso::setmask()
{
    int mask = 0;
    mask |= FixConst::FINAL_INTEGRATE;
    return mask;
}

__global__ void gpu_max_force(
    r64      *force_x,
    r64      *force_y,
    r64      *force_z,
    r64      *mass,
    r64      *max_global,
    const int len )
{
    r64 max_local = 0.;
    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += gridDim.x * blockDim.x ) {
        r64 f = sqrt( force_x[i] * force_x[i] + force_y[i] * force_y[i] + force_z[i] * force_z[i] ) / mass[i];
        max_local = max( max_local, f );
    }
    max_local = __warp_max( max_local );
    if( __laneid() == 0 ) atomicMax( max_global, max_local );
}

__global__ void gpu_fix_opt(
    r64* __restrict coord_x,
    r64* __restrict coord_y,
    r64* __restrict coord_z,
    r64* __restrict veloc_x,
    r64* __restrict veloc_y,
    r64* __restrict veloc_z,
    r64* __restrict force_x,
    r64* __restrict force_y,
    r64* __restrict force_z,
    int* __restrict mask,
    r64* __restrict mass,
    r64* __restrict max_accl,
    const r64  dr,
    const r64  noise,
    const int  groupbit,
    const int  n_atom )
{
    r64 dt2 = 2.0 * dr / *max_accl;
    for( int i = blockDim.x * blockIdx.x + threadIdx.x ; i < n_atom ; i += gridDim.x * blockDim.x ) {
        if( mask[i] & groupbit ) {
            double minv = __rcp( mass[i] );
            coord_x[i] += 0.5 * ( force_x[i] * minv ) * dt2;
            coord_y[i] += 0.5 * ( force_y[i] * minv ) * dt2;
            coord_z[i] += 0.5 * ( force_z[i] * minv ) * dt2;
            veloc_x[i] = 0.;
            veloc_y[i] = 0.;
            veloc_z[i] = 0.;
            if( noise ) {
                uint lcg, key;
                key = __mantissa( force_x[i], force_y[i], force_z[i] );
                lcg = i * 1103515245 + 12345;
                coord_x[i] += noise * uniform_TEA<16>( lcg, key );
                lcg = lcg * 1103515245 + 12345;
                coord_y[i] += noise * uniform_TEA<16>( lcg, key );
                lcg = lcg * 1103515245 + 12345;
                coord_z[i] += noise * uniform_TEA<16>( lcg, key );
            }
        }
    }
}

void FixOptMeso::final_integrate()
{
    static GridConfig grid_cfg, grid_cfg2;
    if( !grid_cfg.x ) {
        grid_cfg = meso_device->occu_calc.right_peak( 0, gpu_fix_opt, 0, cudaFuncCachePreferL1 );
        cudaFuncSetCacheConfig( gpu_fix_opt, cudaFuncCachePreferL1 );
        grid_cfg2 = meso_device->occu_calc.right_peak( 0, gpu_max_force, 0, cudaFuncCachePreferL1 );
        cudaFuncSetCacheConfig( gpu_max_force, cudaFuncCachePreferL1 );
    }
    dev_max_accl.set( 0, meso_device->stream() );
    gpu_max_force <<< grid_cfg2.x, grid_cfg2.y, 0, meso_device->stream() >>> (
        meso_atom->dev_force(0),
        meso_atom->dev_force(1),
        meso_atom->dev_force(2),
        meso_atom->dev_mass,
        dev_max_accl,
        atom->nlocal );
    gpu_fix_opt <<< grid_cfg.x, grid_cfg.y, 0, meso_device->stream() >>> (
        meso_atom->dev_coord(0),
        meso_atom->dev_coord(1),
        meso_atom->dev_coord(2),
        meso_atom->dev_veloc(0),
        meso_atom->dev_veloc(1),
        meso_atom->dev_veloc(2),
        meso_atom->dev_force(0),
        meso_atom->dev_force(1),
        meso_atom->dev_force(2),
        meso_atom->dev_mask,
        meso_atom->dev_mass,
        dev_max_accl,
        max_move,
        noise_level,
        groupbit,
        atom->nlocal );
}
