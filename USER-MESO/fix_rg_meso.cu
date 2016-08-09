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

#include "domain.h"
#include "mpi.h"
#include "stdio.h"
#include "string.h"
#include "force.h"
#include "update.h"
#include "error.h"
#include "pair.h"
#include "neigh_list.h"
#include "group.h"

#include "atom_meso.h"
#include "atom_vec_meso.h"
#include "comm_meso.h"
#include "engine_meso.h"
#include "fix_rg_meso.h"
#include "neighbor_meso.h"
#include "neigh_list_meso.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

MesoFixRg::MesoFixRg( LAMMPS *lmp, int narg, char **arg ):
    Fix( lmp, narg, arg ),
    MesoPointers( lmp ),
    dev_com( lmp, "MesoFixRg::dev_com" ),
    dev_com_mass( lmp, "MesoFixRg::dev_com_n" ),
    dev_rg_sq( lmp, "MesoFixRg::dev_rg" ),
    dev_rg_sum( lmp, "MesoFixRg::dev_rg_mean" ),
    n_mol(0),
    n_smoothing(1),
    c_smoothing(0)
{
    if( narg < 4 ) {
        error->all( FLERR, "<MESO> fix rg/meso usage: output_filename nsmoothing [target=group]" );
    }

    target_group = groupbit;

    for( int i = 0 ; i < narg ; i++ ) {
        if( !strcmp( arg[i], "output" ) ) {
            if( ++i >= narg ) error->all( FLERR, "Incomplete fix rg command after 'output'" );
            output = arg[i];
        } else if( !strcmp( arg[i], "nsmoothing" ) ) {
            if( ++i >= narg ) error->all( FLERR, "Incomplete fix rg command after 'nsmoothing'" );
            n_smoothing = atoi( arg[i] );
        } else if( !strcmp( arg[i], "target" ) ) {
            if( ++i >= narg ) error->all( FLERR, "Incomplete fix rg command after 'target'" );
            target_group = group->bitmask[ group->find( arg[i] ) ];
        }
    }


    scalar_flag = 1;
    extscalar   = 0;
    global_freq = n_smoothing;
    last_scalar = 0.0;

    if ( output.size() ) fout.open( output.c_str() );
}

MesoFixRg::~MesoFixRg()
{
    if ( c_smoothing ) dump();
}

void MesoFixRg::init()
{
    c_smoothing = 0;
    if ( !atom->molecule ) error->all( FLERR, "<MESO> fix rg/meso only works on molecular systems");
    if ( comm->nprocs > 1 ) error->all( FLERR, "<MESO> fix rg/meso only works on a single rank");
}

int MesoFixRg::setmask()
{
    return FixConst::POST_INTEGRATE;
}

__global__ void gpu_fix_rg_com_count(
	int* __restrict molecule,
	r64* __restrict mass,
	r32* __restrict com_mass,
	const int n
)
{
	for( int i = threadIdx.x + blockIdx.x * blockDim.x; i < n ; i += gridDim.x * blockDim.x ) {
		int mol = molecule[i] - 1;
		atomic_add( com_mass + mol, mass[i] );
	}
}

void MesoFixRg::setup( int vflag )
{
	static GridConfig grid_cfg;
	if( !grid_cfg.x ) {
		grid_cfg = meso_device->occu_calc.right_peak( 0, gpu_fix_rg_com_count, 0, cudaFuncCachePreferShared );
	}

	n_mol = 0;
	for(int i=0;i<atom->nlocal;i++) {
		if (atom->mask[i] & groupbit)
			n_mol = std::max( n_mol, atom->molecule[i] );
	}

	dev_com     .grow( n_mol );
	dev_com_mass.grow( n_mol );
	dev_rg_sq   .grow( n_mol );
	dev_rg_sum  .grow( 1 );
	dev_rg_sum.set( 0., meso_device->stream() );

	dev_com_mass.set( 0.f, meso_device->stream() );
	gpu_fix_rg_com_count<<< grid_cfg.x, grid_cfg.y, 0, meso_device->stream() >>> (
		meso_atom->dev_mole,
		meso_atom->dev_mass,
		dev_com_mass,
		atom->nlocal
	);

	post_integrate();
}

__global__ void gpu_fix_rg_com_reduce(
	r64* __restrict coordx,
	r64* __restrict coordy,
	r64* __restrict coordz,
	r64* __restrict mass,
	tagint* __restrict image,
	int* __restrict molecule,
	r32* __restrict comx,
	r32* __restrict comy,
	r32* __restrict comz,
	const r32 box_size_x,
	const r32 box_size_y,
	const r32 box_size_z,
	const int n
)
{
	for( int i = threadIdx.x + blockIdx.x * blockDim.x; i < n ; i += gridDim.x * blockDim.x ) {
		int mol = molecule[i] - 1;
		tagint img = image[i];
		tagint ximg = (  img            & IMGMASK ) - IMGMAX;
		tagint yimg = ( (img>>IMGBITS)  & IMGMASK ) - IMGMAX;
		tagint zimg = ( (img>>IMG2BITS) & IMGMASK ) - IMGMAX;
		atomic_add( comx + mol, ( coordx[i] + ximg * box_size_x ) * mass[i] );
		atomic_add( comy + mol, ( coordy[i] + yimg * box_size_y ) * mass[i] );
		atomic_add( comz + mol, ( coordz[i] + zimg * box_size_z ) * mass[i] );
	}
}

__global__ void gpu_fix_rg_com_mean(
	r32* __restrict comx,
	r32* __restrict comy,
	r32* __restrict comz,
	r32* __restrict com_mass,
	const int n_mol
)
{
	for( int i = threadIdx.x + blockIdx.x * blockDim.x; i < n_mol ; i += gridDim.x * blockDim.x ) {
		r32 m_inv = 1.f / com_mass[i];
		comx[i] *= m_inv;
		comy[i] *= m_inv;
		comz[i] *= m_inv;
	}
}

__global__ void gpu_fix_rg_reduce(
	r64* __restrict coordx,
	r64* __restrict coordy,
	r64* __restrict coordz,
	r64* __restrict mass,
	int* __restrict mask,
	tagint* __restrict image,
	int* __restrict molecule,
	r32* __restrict comx,
	r32* __restrict comy,
	r32* __restrict comz,
	r32* __restrict rg_sq,
	const r32 box_size_x,
	const r32 box_size_y,
	const r32 box_size_z,
        const int target_group,
	const int n
)
{
	for( int i = threadIdx.x + blockIdx.x * blockDim.x; i < n ; i += gridDim.x * blockDim.x ) {
                if ( mask[i] & target_group ) {
		int mol = molecule[i] - 1;
		tagint img = image[i];
		tagint ximg = (  img            & IMGMASK ) - IMGMAX;
		tagint yimg = ( (img>>IMGBITS)  & IMGMASK ) - IMGMAX;
		tagint zimg = ( (img>>IMG2BITS) & IMGMASK ) - IMGMAX;
		r64 dx = ( coordx[i] + ximg * box_size_x ) - comx[mol];
		r64 dy = ( coordy[i] + yimg * box_size_y ) - comy[mol];
		r64 dz = ( coordz[i] + zimg * box_size_z ) - comz[mol];
		r64 rg2 = dx*dx + dy*dy + dz*dz;
		atomic_add( rg_sq + mol, rg2 * mass[i] );
		}
	}
}

__global__ void gpu_fix_rg_mean(
	r32* __restrict rg_sq,
	r64* __restrict rg_sum,
	r32* __restrict com_mass,
	const int n_mol
)
{
	for( int i = threadIdx.x + blockIdx.x * blockDim.x; i < n_mol ; i += gridDim.x * blockDim.x ) {
		//atomic_add( rg_sum, sqrtf( rg_sq[i] / com_mass[i] ) );
		atomic_add( rg_sum, rg_sq[i] / power<2>( com_mass[i] ) ); // for locating T_theta
	}
}

void MesoFixRg::post_integrate()
{
    static GridConfig grid_cfg1, grid_cfg2, grid_cfg3, grid_cfg4;
    if( !grid_cfg1.x ) {
        grid_cfg1 = meso_device->occu_calc.right_peak( 0, gpu_fix_rg_com_reduce, 0, cudaFuncCachePreferShared );
        grid_cfg2 = meso_device->occu_calc.right_peak( 0, gpu_fix_rg_com_mean, 0, cudaFuncCachePreferShared );
        grid_cfg3 = meso_device->occu_calc.right_peak( 0, gpu_fix_rg_reduce, 0, cudaFuncCachePreferShared );
        grid_cfg4 = meso_device->occu_calc.right_peak( 0, gpu_fix_rg_mean, 0, cudaFuncCachePreferShared );
    }

    dev_com.set( 0.f, meso_device->stream() );
    dev_rg_sq.set( 0.f, meso_device->stream() );

    gpu_fix_rg_com_reduce<<< grid_cfg1.x, grid_cfg1.y, 0, meso_device->stream() >>> (
    	meso_atom->dev_coord[0], meso_atom->dev_coord[1], meso_atom->dev_coord[2],
    	meso_atom->dev_mass,
    	meso_atom->dev_image,
    	meso_atom->dev_mole,
    	dev_com[0],
    	dev_com[1],
    	dev_com[2],
    	domain->xprd, domain->yprd, domain->zprd,
    	atom->nlocal );

    gpu_fix_rg_com_mean<<< grid_cfg2.x, grid_cfg2.y, 0, meso_device->stream() >>> (
    	dev_com[0],
    	dev_com[1],
    	dev_com[2],
    	dev_com_mass,
    	n_mol );

    gpu_fix_rg_reduce<<< grid_cfg3.x, grid_cfg3.y, 0, meso_device->stream() >>> (
    	meso_atom->dev_coord[0], meso_atom->dev_coord[1], meso_atom->dev_coord[2],
    	meso_atom->dev_mass,
    	meso_atom->dev_mask,
    	meso_atom->dev_image,
    	meso_atom->dev_mole,
    	dev_com[0],
    	dev_com[1],
    	dev_com[2],
    	dev_rg_sq,
    	domain->xprd, domain->yprd, domain->zprd,
	target_group,
    	atom->nlocal );

    gpu_fix_rg_mean<<< grid_cfg4.x, grid_cfg4.y, 0, meso_device->stream() >>> (
    	dev_rg_sq,
    	dev_rg_sum,
    	dev_com_mass,
    	n_mol );

    if ( ++c_smoothing == n_smoothing ) dump();
}

void MesoFixRg::dump()
{
    // dump result
	double rg_sum;
	dev_rg_sum.download( &rg_sum, dev_rg_sum.n(), meso_device->stream() );
	dev_rg_sum.set( 0., meso_device->stream() );
    meso_device->sync_device();

    rg_sum /= ( c_smoothing * n_mol );
    if ( output.size() ) fout << rg_sum << std::endl;
    last_scalar = rg_sum;
    c_smoothing = 0;
}

double MesoFixRg::compute_scalar() {
	return last_scalar;
}

