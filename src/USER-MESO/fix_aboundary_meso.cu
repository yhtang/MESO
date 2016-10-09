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
	nevery        = 1;
	wall_group    = -1;
	wall_groupbit = 0;
    rc            = 0;
    rho0          = 0;
    sigma         = 0;

    for(int i = 0 ; i < narg ; i++) {
    	if (!strcmp(arg[i],"wall")) {
			if ( ( wall_group = group->find( arg[++i] ) ) == -1 ) {
				error->all( FLERR, "<MESO> Undefined wall group id in fix aboundary/meso" );
			}
			wall_groupbit = group->bitmask[ wall_group ];
			if ( groupbit & wall_groupbit ) {
				error->warning( FLERR, "<MESO> fix aboundary/meso mobile and wall group overlapped", true );
			}
    	}
    	if (!strcmp(arg[i],"rc")) {
    		rc = atof( arg[++i] );
    		continue;
    	}
    	if (!strcmp(arg[i],"rho")) {
    		rho0 = atof( arg[++i] );
    		continue;
    	}
    	if (!strcmp(arg[i],"sigma")) {
    		sigma = atof( arg[++i] );
    		continue;
    	}
    }

    if ( wall_groupbit == 0 || rc == 0 || rho0 == 0 ) {
		std::stringstream msg;
		msg << "<MESO> Fix aboundary/meso usage: " << arg[2]
			<< " <wall groupid> <rc double> <rho double> [sigma double]"
			<< std::endl;
		error->all( __FILE__, __LINE__, msg.str().c_str() );
    }

    /* Gaussian Kernel
     * p(x) = 1 / sqrt(2pi)^D / sqrt(sigma^D) * exp( -1/2 * (x-u)^T Sigma^-1 (x-u) )
     */

    if ( sigma == 0 ) sigma = rc * rc / 4.0 / M_PI;
}

int FixArbitraryBoundary::setmask()
{
    int mask = 0;
    //mask |= FixConst::POST_FORCE;
    mask |= FixConst::END_OF_STEP;
    return mask;
}

__global__ void gpu_aboundary(
    texobj tex_coord, texobj tex_mask, texobj tex_tag,
    r64* __restrict coord_x,   r64* __restrict coord_y,   r64* __restrict coord_z,
    r64* __restrict veloc_x,   r64* __restrict veloc_y,   r64* __restrict veloc_z,
    int* __restrict pair_count, int* __restrict pair_table,
    const r32 rho0,
    const r32 nfactor,
    const r32 sigma,
    const r32 rc,
    const int pair_padding,
    const int mobile_bit,
    const int wall_bit,
	const int n )
{
    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x ) {
    	if ( !(tex1Dfetch<int>( tex_mask, i ) & mobile_bit) ) continue;

    	f3u  coord1 = tex1Dfetch<float4>( tex_coord, i ); // must fetch from texture, because shifted by domain local center
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
			r32 depth = min( rc, sqrtf(2.f) * sqrtf(sigma) * erfinvf( 2.f * rho_wall / rho0 - 1.f ) );
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
		meso_atom->dev_coord(0), meso_atom->dev_coord(1), meso_atom->dev_coord(2),
		meso_atom->dev_veloc(0), meso_atom->dev_veloc(1), meso_atom->dev_veloc(2),
        list->dev_pair_count_core,
        list->dev_pair_table,
        rho0,
        1 / sqrt( pow( 2*M_PI, 3. ) * pow( sigma, 3. ) ),
        sigma,
        rc,
        list->n_col,
        groupbit,
        wall_groupbit,
        atom->nlocal );
}

__global__ void gpu_aboundary_v2(
    texobj tex_coord, texobj tex_mask,
    r64* __restrict coord_x,   r64* __restrict coord_y,   r64* __restrict coord_z,
    r64* __restrict veloc_x,   r64* __restrict veloc_y,   r64* __restrict veloc_z,
    r64* __restrict force_x,   r64* __restrict force_y,   r64* __restrict force_z,
    int* __restrict pair_count, int* __restrict pair_table,
    const r32 rho0,
    const r32 nfactor,
    const r32 sigma,
    const r32 rc,
    const r32 dt,
    const int pair_padding,
    const int mobile_bit,
    const int wall_bit,
	const int n )
{
	for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x ) {
    	if ( !(tex1Dfetch<int>( tex_mask, i ) & mobile_bit) ) continue;

    	// project velocity and position
    	const f3u    tmp = tex1Dfetch<float4>( tex_coord, i ); // must fetch from texture, because shifted by domain local center
    	const float3 x0 = make_float3( tmp.x, tmp.y, tmp.z );
    	const float3 v0 = make_float3( veloc_x[i], veloc_y[i], veloc_z[i] );
    	const float3 f0 = make_float3( force_x[i], force_y[i], force_z[i] );
    	const float3 v  = v0 + 0.5 * dt * f0;
    	const float3 coord1 = x0 + dt * v;

		int  n_pair = pair_count[i];
		int *p_pair = pair_table + ( i - __laneid() ) * pair_padding + __laneid();
		float   rho_wall = 0.f;
		float3  drho_wall = make_float3( 0.f, 0.f, 0.f );

		// compute density using projected position
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
				rho_wall    += w;
				drho_wall    = drho_wall + make_float3( dx, dy, dz ) * ( w / sigma );
			}
		}

		// if will be entering wall: invert normal velocity
		if ( rho_wall > 0.5f * rho0 ) {
			const float3 e = -normalize( drho_wall );
			const float vdote = dot( v0, e );
			const float fdote = dot( f0, e );
			veloc_x[i] -= 2.f * max( 0.f, vdote ) * e.x;
			veloc_y[i] -= 2.f * max( 0.f, vdote ) * e.y;
			veloc_z[i] -= 2.f * max( 0.f, vdote ) * e.z;
			force_x[i] -= 2.f * max( 0.f, fdote ) * e.x;
			force_y[i] -= 2.f * max( 0.f, fdote ) * e.y;
			force_z[i] -= 2.f * max( 0.f, fdote ) * e.z;
		}
    }
}

void FixArbitraryBoundary::end_of_step()
{
	MesoNeighList *list = meso_neighbor->lists_device[ force->pair->list->index ];
    if( !list ) error->all( FLERR, "<MESO> fix aboundary/meso must be used together with a pair from USER-MESO" );

    meso_atom->meso_avec->dp2sp_merged( 0, 0, atom->nlocal+atom->nghost, true );
    meso_atom->tex_misc("mask").bind( *(meso_atom->dev_mask) );

    static GridConfig grid_cfg;
    if (!grid_cfg.x) {
    	grid_cfg = meso_device->configure_kernel( gpu_aboundary_v2 );
    }

	gpu_aboundary_v2<<< grid_cfg.x, grid_cfg.y, 0, meso_device->stream() >>> (
		meso_atom->tex_coord_merged,
		meso_atom->tex_misc("mask"),
		meso_atom->dev_coord(0), meso_atom->dev_coord(1), meso_atom->dev_coord(2),
		meso_atom->dev_veloc(0), meso_atom->dev_veloc(1), meso_atom->dev_veloc(2),
		meso_atom->dev_force(0), meso_atom->dev_force(1), meso_atom->dev_force(2),
        list->dev_pair_count_core,
        list->dev_pair_table,
        rho0,
        1 / sqrt( pow( 2*M_PI, 3. ) * pow( sigma, 3. ) ),
        sigma,
        rc,
        update->dt,
        list->n_col,
        groupbit,
        wall_groupbit,
        atom->nlocal );
}
