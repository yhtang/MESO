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
#include "fix_nve_meso.h"
#include "engine_meso.h"
#include "atom_meso.h"
#include "comm_meso.h"

using namespace LAMMPS_NS;

FixNVEMeso::FixNVEMeso( LAMMPS *lmp, int narg, char **arg ) :
    Fix( lmp, narg, arg ),
    MesoPointers( lmp ),
    dtv( 0 ), dtf( 0 ), dtT( 0 )
{
    time_integrate = 1;

    if ( atom->T && atom->Q ) {
		if ( narg < 4 ) error->all( FLERR, "Invalid nve/meso command, usage: Id group style Cv" );
		Cv = atof( arg[3] );
    }
}

void FixNVEMeso::init()
{
    dtv = update->dt;
    dtf = 0.5 * update->dt * force->ftm2v;
    dtT = update->dt / Cv;

#ifdef LMP_MESO_LOG_L1
    if( strcmp( update->integrate_style, "respa" ) == 0 )
        fprintf( stderr, "<MESO> respa style directive detected, RESPA not supported in CUDA at present.\n" );
#endif
}

int FixNVEMeso::setmask()
{
    int mask = 0;
    mask |= FixConst::INITIAL_INTEGRATE;
    mask |= FixConst::FINAL_INTEGRATE;
    return mask;
}

template<int EDPD>
__global__ void gpu_fix_NVE_init_intgrate(
    r64* __restrict coord_x,
    r64* __restrict coord_y,
    r64* __restrict coord_z,
    r64* __restrict veloc_x,
    r64* __restrict veloc_y,
    r64* __restrict veloc_z,
    r64* __restrict force_x,
    r64* __restrict force_y,
    r64* __restrict force_z,
    r64* __restrict T,
    r64* __restrict Q,
    int* __restrict mask,
    r64* __restrict mass,
    const r64  dtf,
    const r64  dtv,
    const r64  dtT,
    const int  groupbit,
    const int  n_atom )
{
    for( int i = blockDim.x * blockIdx.x + threadIdx.x ; i < n_atom ; i += gridDim.x * blockDim.x ) {
        if( mask[i] & groupbit ) {
            r64 dtfm = dtf * __rcp( mass[i] );
            veloc_x [i] += dtfm * force_x[i];
            veloc_y [i] += dtfm * force_y[i];
            veloc_z [i] += dtfm * force_z[i];
            coord_x[i]  += dtv  * veloc_x[i];
            coord_y[i]  += dtv  * veloc_y[i];
            coord_z[i]  += dtv  * veloc_z[i];
            if( EDPD ) T[i] += Q[i] * dtT ;
        }
    }
}

void FixNVEMeso::initial_integrate( __attribute__( ( unused ) ) int vflag )
{
//	printf("%s %d ",__FILE__,__LINE__);
//	meso_device->sync_device();
//	check_acc<<<1,1>>>( meso_atom->dev_force(0), meso_atom->dev_force(1), meso_atom->dev_force(2), atom->nlocal );
//	meso_device->sync_device();

    if( atom->T && atom->Q ) {  // EDPD
        static GridConfig grid_cfg;
        if( !grid_cfg.x ) {
            grid_cfg = meso_device->occu_calc.right_peak( 0, gpu_fix_NVE_init_intgrate<1>, 0, cudaFuncCachePreferL1 );
            cudaFuncSetCacheConfig( gpu_fix_NVE_init_intgrate<1>, cudaFuncCachePreferL1 );
        }
        gpu_fix_NVE_init_intgrate<1> <<< grid_cfg.x, grid_cfg.y, 0, meso_device->stream() >>> (
            meso_atom->dev_coord(0),
            meso_atom->dev_coord(1),
            meso_atom->dev_coord(2),
            meso_atom->dev_veloc(0),
            meso_atom->dev_veloc(1),
            meso_atom->dev_veloc(2),
            meso_atom->dev_force(0),
            meso_atom->dev_force(1),
            meso_atom->dev_force(2),
            meso_atom->dev_T,
            meso_atom->dev_Q,
            meso_atom->dev_mask,
            meso_atom->dev_mass,
            dtf,
            dtv,
            dtT,
            groupbit,
            atom->nlocal );
    } else { // classical DPD
        static GridConfig grid_cfg;
        if( !grid_cfg.x ) {
            grid_cfg = meso_device->occu_calc.right_peak( 0, gpu_fix_NVE_init_intgrate<0>, 0, cudaFuncCachePreferL1 );
            cudaFuncSetCacheConfig( gpu_fix_NVE_init_intgrate<0>, cudaFuncCachePreferL1 );
        }
        gpu_fix_NVE_init_intgrate<0> <<< grid_cfg.x, grid_cfg.y, 0, meso_device->stream() >>> (
            meso_atom->dev_coord(0),
            meso_atom->dev_coord(1),
            meso_atom->dev_coord(2),
            meso_atom->dev_veloc(0),
            meso_atom->dev_veloc(1),
            meso_atom->dev_veloc(2),
            meso_atom->dev_force(0),
            meso_atom->dev_force(1),
            meso_atom->dev_force(2),
            NULL,
            NULL,
            meso_atom->dev_mask,
            meso_atom->dev_mass,
            dtf,
            dtv,
            0,
            groupbit,
            atom->nlocal );
    }
}

__global__ void gpu_fix_NVE_final_integrate(
    r64* __restrict veloc_x,
    r64* __restrict veloc_y,
    r64* __restrict veloc_z,
    r64* __restrict force_x,
    r64* __restrict force_y,
    r64* __restrict force_z,
    int* __restrict mask,
    r64* __restrict mass,
    const r64  dtf,
    const int  groupbit,
    const int  n_atom )
{
    for( int i = blockDim.x * blockIdx.x + threadIdx.x ; i < n_atom ; i += gridDim.x * blockDim.x ) {
        if( mask[i] & groupbit ) {
            r64 dtfm = dtf * __rcp( mass[i] );
            veloc_x[i] += dtfm * force_x[i];
            veloc_y[i] += dtfm * force_y[i];
            veloc_z[i] += dtfm * force_z[i];
        }
    }
}

void FixNVEMeso::final_integrate()
{
    static GridConfig grid_cfg;
    if( !grid_cfg.x ) {
        grid_cfg = meso_device->occu_calc.right_peak( 0, gpu_fix_NVE_final_integrate, 0, cudaFuncCachePreferL1 );
        cudaFuncSetCacheConfig( gpu_fix_NVE_final_integrate, cudaFuncCachePreferL1 );
    }
    gpu_fix_NVE_final_integrate <<< grid_cfg.x, grid_cfg.y, 0, meso_device->stream() >>> (
        meso_atom->dev_veloc(0),
        meso_atom->dev_veloc(1),
        meso_atom->dev_veloc(2),
        meso_atom->dev_force(0),
        meso_atom->dev_force(1),
        meso_atom->dev_force(2),
        meso_atom->dev_mask,
        meso_atom->dev_mass,
        dtf,
        groupbit,
        atom->nlocal );
}

void FixNVEMeso::reset_dt()
{
    dtv = update->dt;
    dtf = 0.5 * update->dt * force->ftm2v;
}
