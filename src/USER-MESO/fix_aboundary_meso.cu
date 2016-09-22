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
#include "force.h"
#include "update.h"
#include "neigh_list.h"
#include "error.h"
#include "group.h"

#include "atom_meso.h"
#include "atom_vec_meso.h"
#include "comm_meso.h"
#include "engine_meso.h"
#include "neighbor_meso.h"
#include "neigh_list_meso.h"
#include "pair_edpd_trp_meso.h"
#include "fix_aboundary_meso.h"

using namespace LAMMPS_NS;
using namespace PNIPAM_COEFFICIENTS;

FixArbitraryBoundary::FixArbitraryBoundary( LAMMPS *lmp, int narg, char **arg ) :
    Fix( lmp, narg, arg ),
    MesoPointers( lmp )
{
    if( narg < 5 ) {
    	std::stringstream msg;
    	msg << "<MESO> Fix aboundary/meso usage: " << arg[2]
    	    << " wall_group rc [sigma double] [rho double]"
    	    << std::endl;
    	error->all( __FILE__, __LINE__, msg.str().c_str() );
    }

    if ( ( wall_group = group->find( arg[3] ) ) == -1 ) {
    	error->all( FLERR, "<MESO> Undefined group id in fix aboundary/meso" );
    }
    wall_groupbit = group->bitmask[ wall_group ];
    if ( groupbit & wall_groupbit ) {
    	error->warning( FLERR, "<MESO> fix aboundary/meso mobile and wall group overlapped", true );
    }

    rc   = atof( arg[4] );
    rho0 = 0;
    sigma = rc * rc / 4.0 / M_PI;

    for(int i=5;i<narg;i++) {
    	if (!strcmp(arg[i],"sigma")) {
    		sigma = atof( arg[++i] );
    		continue;
    	}
    	if (!strcmp(arg[i],"rho")) {
    		rho0 = atof( arg[++i] );
    		continue;
    	}
    }
}

int FixArbitraryBoundary::setmask()
{
    int mask = 0;
    mask |= FixConst::POST_FORCE;
    return mask;
}

/* Gaussian Kernel
 * p(x) = 1 / sqrt(2pi)^D / sqrt(sigma^D) * exp( -1/2 * (x-u)^T Sigma^-1 (x-u) )
 */

__global__ void gpu_find_rho0(
    texobj tex_coord, texobj tex_mask,
    float* __restrict max_rho0,
    int* __restrict pair_count, int* __restrict pair_table,
    const r32 nfactor,
    const r32 sigma,
    const int pair_padding,
    const int wall_bit,
	const int nlocal )
{
    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < nlocal; i += gridDim.x * blockDim.x ) {
    	if ( !(tex1Dfetch<int>( tex_mask, i ) & wall_bit) ) continue;

    	f3u  coord1 = tex1Dfetch<float4>( tex_coord, i );
		int  n_pair = pair_count[i];
		int *p_pair = pair_table + ( i - __laneid() ) * pair_padding + __laneid();
		float rho = 0.f;
		float3 drho = make_float3( 0.f, 0.f, 0.f );

		for( int p = 0; p < n_pair; p++ ) {
			int j   = __lds( p_pair );
			p_pair += pair_padding;
			if( ( p & (WARPSZ - 1) ) == WARPSZ - 1 ) p_pair -= WARPSZ * pair_padding - WARPSZ;
			if ( j < nlocal && (tex1Dfetch<int>( tex_mask, j ) & wall_bit) )  {
				f3u coord2   = tex1Dfetch<float4>( tex_coord, j );
				r32 dx       = coord1.x - coord2.x;
				r32 dy       = coord1.y - coord2.y;
				r32 dz       = coord1.z - coord2.z;
				r32 rsq      = dx * dx + dy * dy + dz * dz;
				r32 w        = nfactor * expf( -0.5f / sigma * rsq );
				rho         += w;
				drho         = drho + make_float3( dx, dy, dz ) * ( w / sigma );
			}
		}

		for( int i = 1; i < 32; i <<= 1 ) rho = max( rho, __shfl_xor( rho, i ) );
		if ( __laneid() == 0 ) atomicMax( max_rho0, rho );
    }
}

__global__ void gpu_aboundary(
    texobj tex_coord, texobj tex_mask, texobj tex_tag,
    r64* __restrict coord_x,   r64* __restrict coord_y,   r64* __restrict coord_z,
    r64* __restrict veloc_x,   r64* __restrict veloc_y,   r64* __restrict veloc_z,
    int* __restrict pair_count, int* __restrict pair_table,
    const r32 rho0,
    const r32 nfactor,
    const r32 sigma,
    const int pair_padding,
    const int mobile_bit,
    const int wall_bit,
	const int n )
{
    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x ) {
    	if ( !(tex1Dfetch<int>( tex_mask, i ) & mobile_bit) ) continue;

    	f3u  coord1 = tex1Dfetch<float4>( tex_coord, i );
		int  n_pair = pair_count[i];
		int *p_pair = pair_table + ( i - __laneid() ) * pair_padding + __laneid();
		float   rho_wall = 0.f;
		float3  drho_wall = make_float3( 0.f, 0.f, 0.f );

		for( int p = 0; p < n_pair; p++ ) {
			int j   = __lds( p_pair );
			p_pair += pair_padding;
			if( ( p & (WARPSZ - 1) ) == WARPSZ - 1 ) p_pair -= WARPSZ * pair_padding - WARPSZ;
			if ( tex1Dfetch<int>( tex_mask, j ) & wall_bit )  {
				f3u coord2   = tex1Dfetch<float4>( tex_coord, j );
				r32 dx       = coord1.x - coord2.x;
				r32 dy       = coord1.y - coord2.y;
				r32 dz       = coord1.z - coord2.z;
				r32 rsq      = dx * dx + dy * dy + dz * dz;

				r32 w        = nfactor * expf( -0.5f / sigma * rsq );
				rho_wall += w;
				drho_wall = drho_wall + make_float3( dx, dy, dz ) * ( w / sigma );
			}
		}

		if ( rho_wall > 0.5f * rho0 ) {
			r32 depth = sqrtf(2.f) * sqrtf(sigma) * erfinvf( 2.f * min( rho_wall / rho0, .9999f ) - 1.f );
			float3 x = make_float3( coord_x[i], coord_y[i], coord_z[i] );
			float3 e = -normalize( drho_wall );
			float3 x_new = x - 2.f * depth * e;
			coord_x[i] = x_new.x; coord_y[i] = x_new.y; coord_z[i] = x_new.z;
			veloc_x[i] = -veloc_x[i]; veloc_y[i] = -veloc_y[i]; veloc_z[i] = -veloc_z[i];
		}
    }
}

void FixArbitraryBoundary::post_force(int evflag)
{
	if (!rho0) rho0 = find_rho0();

	MesoNeighList *list = meso_neighbor->lists_device[ force->pair->list->index ];
    if( !list ) error->all( FLERR, "<MESO> fix aboundary/meso must be used together with a pair from USER-MESO" );

    meso_atom->meso_avec->dp2sp_merged( 0, 0, atom->nlocal+atom->nghost, true );
	meso_atom->tex_misc("mask").bind( *(meso_atom->dev_mask) );

    static GridConfig grid_cfg;
    if (!grid_cfg.x) {
    	grid_cfg = meso_device->configure_kernel( gpu_aboundary );
    }

	gpu_aboundary<<< grid_cfg.x, grid_cfg.y, 0, meso_device->stream() >>> (
		meso_atom->tex_coord_merged,
		meso_atom->tex_misc("mask"),
		meso_atom->tex_tag,
		meso_atom->dev_coord[0], meso_atom->dev_coord[1], meso_atom->dev_coord[2],
		meso_atom->dev_veloc[0], meso_atom->dev_veloc[1], meso_atom->dev_veloc[2],
        list->dev_pair_count_core,
        list->dev_pair_table,
        rho0,
        1 / sqrt( pow( 2*M_PI, 3. ) * pow( sigma, 3. ) ),
        sigma,
        list->n_col,
        groupbit,
        wall_groupbit,
        atom->nlocal );
}

double FixArbitraryBoundary::find_rho0() {
	DeviceScalar<float> dev_rho0( lmp, "FixArbitraryBoundary::dev_rho0" );
	dev_rho0.grow(1,false,true);

	MesoNeighList *list = meso_neighbor->lists_device[ force->pair->list->index ];
    if( !list ) error->all( FLERR, "<MESO> fix aboundary/meso must be used together with a pair from USER-MESO" );

    meso_atom->meso_avec->dp2sp_merged( 0, 0, atom->nlocal+atom->nghost, true );
	meso_atom->tex_misc("mask").bind( *(meso_atom->dev_mask) );

    static GridConfig grid_cfg;
    if (!grid_cfg.x) {
    	grid_cfg = meso_device->configure_kernel( gpu_find_rho0 );
    }

    gpu_find_rho0<<< grid_cfg.x, grid_cfg.y, 0, meso_device->stream() >>> (
		meso_atom->tex_coord_merged,
		meso_atom->tex_misc("mask"),
		dev_rho0,
		list->dev_pair_count_core,
        list->dev_pair_table,
        1 / sqrt( pow( 2*M_PI, 3. ) * pow( sigma, 3. ) ),
        sigma,
        list->n_col,
        wall_groupbit,
        atom->nlocal );

	float rho;
	dev_rho0.download( &rho, 1 );
	MPI_Allreduce( MPI_IN_PLACE, &rho, 1, MPI_FLOAT, MPI_MAX, world );
	return rho;
}
