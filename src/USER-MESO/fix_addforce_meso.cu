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
#include "fix_addforce_meso.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

MesoFixAddForce::MesoFixAddForce( LAMMPS *lmp, int narg, char **arg ):
    Fix( lmp, narg, arg ),
    MesoPointers( lmp )
{
    if( narg < 6 ) error->all( FLERR, "Illegal fix addforce/meso command" );

    int parg = 3;
    fx = atof( arg[parg++] );
    fy = atof( arg[parg++] );
    fz = atof( arg[parg++] );
}

int MesoFixAddForce::setmask()
{
    int mask = 0;
    mask |= FixConst::POST_FORCE;
    return mask;
}

void MesoFixAddForce::init()
{
    if( strcmp( update->integrate_style, "respa" ) == 0 ) {
        fprintf( stderr, "<MESO> RESPA not supported in MesoFixAddForce. %s %d\n", __FILE__, __LINE__ );
    }
}

void MesoFixAddForce::setup( int vflag )
{
    if( strcmp( update->integrate_style, "respa" ) == 0 ) {
        fprintf( stderr, "<MESO> RESPA not supported in MesoFixAddForce. %s %d\n", __FILE__, __LINE__ );
    }
    post_force( vflag );
}

__global__ void gpu_fix_add_force(
	r64* __restrict force_x,
	r64* __restrict force_y,
	r64* __restrict force_z,
    int* __restrict mask,
    const r64 fx,
    const r64 fy,
    const r64 fz,
    const int groupbit,
    const int n )
{
    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x ) {
        if( mask[i] & groupbit ) {
        	force_x[i] += fx;
        	force_y[i] += fy;
        	force_z[i] += fz;
        }
    }
}

void MesoFixAddForce::post_force( int vflag )
{
    static GridConfig grid_cfg;
    if( !grid_cfg.x ) {
        grid_cfg = meso_device->occu_calc.right_peak( 0, gpu_fix_add_force, 0, cudaFuncCachePreferL1 );
        cudaFuncSetCacheConfig( gpu_fix_add_force, cudaFuncCachePreferL1 );
    }

    gpu_fix_add_force <<< grid_cfg.x, grid_cfg.y, 0, meso_device->stream() >>> (
		meso_atom->dev_force(0),
		meso_atom->dev_force(1),
		meso_atom->dev_force(2),
        meso_atom->dev_mask,
        fx, fy, fz,
        groupbit,
        atom->nlocal );
}
