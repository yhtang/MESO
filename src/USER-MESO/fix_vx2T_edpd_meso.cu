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
#include "mpi.h"
#include "stdio.h"
#include "string.h"
#include "force.h"
#include "update.h"
#include "respa.h"
#include "error.h"

#include "atom_vec_meso.h"
#include "fix_vx2T_edpd_meso.h"
#include "engine_meso.h"
#include "atom_meso.h"
#include "comm_meso.h"

using namespace LAMMPS_NS;

FixEDPDVx2TMeso::FixEDPDVx2TMeso( LAMMPS *lmp, int narg, char **arg ) :
    Fix( lmp, narg, arg ),
    MesoPointers( lmp )
{
    if ( !atom->T ) {
		error->all( FLERR, "Invalid vx2T/edpd/meso command: must be used with eDPD atom vectors" );
    }
}

int FixEDPDVx2TMeso::setmask() {
    return FixConst::PRE_NEIGHBOR;
}

__global__ void gpu_vx2T(
    r64* __restrict veloc_x,
    r64* __restrict T,
    int* __restrict mask,
    const int  groupbit,
    const int  n_atom )
{
    for( int i = blockDim.x * blockIdx.x + threadIdx.x ; i < n_atom ; i += gridDim.x * blockDim.x ) {
        if( mask[i] & groupbit ) {
            T[i] = veloc_x[i];
        }
    }
}

void FixEDPDVx2TMeso::setup_pre_neighbor()
{
	static GridConfig grid_cfg;
	if( !grid_cfg.x ) {
		grid_cfg = meso_device->occu_calc.right_peak( 0, gpu_vx2T, 0, cudaFuncCachePreferL1 );
		cudaFuncSetCacheConfig( gpu_vx2T, cudaFuncCachePreferL1 );
	}
	gpu_vx2T<<< grid_cfg.x, grid_cfg.y, 0, meso_device->stream() >>> (
		meso_atom->dev_veloc(0),
		meso_atom->dev_T,
		meso_atom->dev_mask,
		groupbit,
		atom->nlocal );
}
