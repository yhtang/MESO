/* ----------------------------------------------------------------------
     LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
     http://lammps.sandia.gov, Sandia National Laboratories
     Steve Plimpton, sjplimp@sandia.gov

     Copyright (2003) Sandia Corporation.   Under the terms of Contract
     DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
     certain rights in this software.   This software is distributed under
     the GNU General Public License.

     See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* Exerting force to induce the periodic Poiseuille flow
 * Adapted from the CPU version fix_zl_force first written by:
 * Zhen Li, Crunch Group, Division of Applied Mathematics, Brown University
 * June, 2013
 */

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
#include "fix_poiseuille_meso.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

MesoFixPoiseuille::MesoFixPoiseuille( LAMMPS *lmp, int narg, char **arg ):
    Fix( lmp, narg, arg ),
    MesoPointers( lmp )
{
    if( narg < 6 ) error->all( FLERR, "Illegal fix CUDAPoiseuille command" );

    std::map<char, int> parser;
    parser[ 'x' ] = 0;
    parser[ 'y' ] = 1;
    parser[ 'z' ] = 2;

    if ( std::isdigit(arg[3][0]) ) dim_ortho = atoi( arg[3] );
    else dim_ortho = parser[ arg[3][0] ];
    if ( std::isdigit(arg[4][0]) ) dim_force = atoi( arg[4] );
    else dim_force = parser[ arg[4][0] ];
    strength  = atof( arg[5] );
    if( narg >= 6 )
        bisect_frac = atof( arg[6] );
    else
        bisect_frac = 0.;
}

int MesoFixPoiseuille::setmask()
{
    int mask = 0;
    mask |= FixConst::POST_FORCE;
    mask |= FixConst::MIN_POST_FORCE;
    return mask;
}

void MesoFixPoiseuille::init()
{
    if( strcmp( update->integrate_style, "respa" ) == 0 ) {
        fprintf( stderr, "<MESO> RESPA not supported in MesoFixPoiseuille. %s %d\n", __FILE__, __LINE__ );
    }
}

void MesoFixPoiseuille::setup( int vflag )
{
    if( strcmp( update->integrate_style, "respa" ) == 0 ) {
        fprintf( stderr, "<MESO> RESPA not supported in MesoFixPoiseuille. %s %d\n", __FILE__, __LINE__ );
    }
    post_force( vflag );
}

__global__ void gpu_fix_pois_post_force(
    r64* __restrict coord_ortho,
    r64* __restrict force,
    int* __restrict mask,
    const r64 strength,
    const r64 bisect_point,
    const r64 lower_point,
    const r64 upper_point,
    const int groupbit,
    const int n_work )
{
    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_work; i += gridDim.x * blockDim.x ) {
        if( mask[i] & groupbit ) {
            r64 r = coord_ortho[i];
            if( ( r < bisect_point && r >= lower_point ) || r >= upper_point )
            	force[i] += strength;
            else
            	force[i] -= strength;
        }
    }
}

void MesoFixPoiseuille::post_force( int vflag )
{
    static GridConfig grid_cfg;
    if( !grid_cfg.x ) {
        grid_cfg = meso_device->occu_calc.right_peak( 0, gpu_fix_pois_post_force, 0, cudaFuncCachePreferL1 );
        cudaFuncSetCacheConfig( gpu_fix_pois_post_force, cudaFuncCachePreferL1 );
    }

    r64 bisect_point = bisect_frac * domain->boxhi[dim_ortho] + ( 1.0 - bisect_frac ) * domain->boxlo[dim_ortho];
    gpu_fix_pois_post_force <<< grid_cfg.x, grid_cfg.y, ( grid_cfg.y / WARPSZ ) * 2 * sizeof( r64 ), meso_device->stream() >>> (
        meso_atom->dev_coord[dim_ortho],
        meso_atom->dev_force[dim_force],
        meso_atom->dev_mask,
        strength,
        bisect_point,
        domain->boxlo[dim_ortho],
        domain->boxhi[dim_ortho],
        groupbit,
        atom->nlocal );
}

