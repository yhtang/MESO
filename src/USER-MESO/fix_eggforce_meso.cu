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
#include "fix_eggforce_meso.h"

using namespace LAMMPS_NS;
using namespace PNIPAM_COEFFICIENTS;

FixEggForce::FixEggForce( LAMMPS *lmp, int narg, char **arg ) :
    Fix( lmp, narg, arg ),
    MesoPointers( lmp )
{
    if( narg < 7 ) {
    	std::stringstream msg;
    	msg << "<MESO> Fix eggforce/meso usage: " << arg[2]
    	    << " bodyforce ref_group rho0 rc [sigma double]"
    	    << std::endl;
    	error->all( __FILE__, __LINE__, msg.str().c_str() );
    }

    bodyforce = atof( arg[3] );

    if ( ( ref_group = group->find( arg[4] ) ) == -1 ) {
    	error->all( FLERR, "<MESO> Undefined group id in fix aboundary/meso" );
    }
    ref_groupbit = group->bitmask[ ref_group ];

    rho0 = atof( arg[5] );
    rc   = atof( arg[6] );
    sigma = rc * rc / 4.0 / M_PI;

    for(int i=7;i<narg;i++) {
    	if (!strcmp(arg[i],"sigma")) {
    		sigma = atof( arg[++i] );
    		continue;
    	}
    }
}

int FixEggForce::setmask()
{
    int mask = 0;
    mask |= FixConst::POST_FORCE;
    return mask;
}

/* Gaussian Kernel
 * p(x) = 1 / sqrt(2pi)^D / sqrt(sigma^D) * exp( -1/2 * (x-u)^T Sigma^-1 (x-u) )
 */

__global__ void gpu_eggforce(
	texobj tex_coord, texobj tex_veloc, texobj tex_mask, texobj tex_tag,
    r64* __restrict force_x,   r64* __restrict force_y,   r64* __restrict force_z,
    int* __restrict pair_count, int* __restrict pair_table,
    const r32 rho0,
    const r32 bodyforce,
    const r32 nfactor,
    const r32 sigma,
    const int pair_padding,
    const int mobile_bit,
    const int ref_bit,
	const int n )
{
    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x ) {
    	if ( !(tex1Dfetch<int>( tex_mask, i ) & mobile_bit) ) continue;

    	f3u  coord1 = tex1Dfetch<float4>( tex_coord, i );
		int  n_pair = pair_count[i];
		int *p_pair = pair_table + ( i - __laneid() ) * pair_padding + __laneid();
		float3 v = make_float3(0,0,0);
		float rho = 0.f;
		float3 drho = make_float3( 0.f, 0.f, 0.f );

		for( int p = 0; p < n_pair; p++ ) {
			int j   = __lds( p_pair );
			p_pair += pair_padding;
			if( ( p & (WARPSZ - 1) ) == WARPSZ - 1 ) p_pair -= WARPSZ * pair_padding - WARPSZ;
			if ( !(tex1Dfetch<int>( tex_mask, j ) & ref_bit) ) continue;

			f3u coord2   = tex1Dfetch<float4>( tex_coord, j );
			f3u veloc2   = tex1Dfetch<float4>( tex_veloc, j );
			r32 dx       = coord1.x - coord2.x;
			r32 dy       = coord1.y - coord2.y;
			r32 dz       = coord1.z - coord2.z;
			r32 rsq      = dx * dx + dy * dy + dz * dz;

			r32 w        = nfactor * expf( -0.5f / sigma * rsq );
			v            = v + make_float3(veloc2.x,veloc2.y,veloc2.z) * w;
			rho         += w;
			drho         = drho + make_float3( dx, dy, dz ) * ( w / sigma );
		}

		if ( rho > rho0 * 0.5 ) {
#if 0
			if ( rho > rho0 * 0.9 ) {
				v = normalize(v);
			} else {
				drho = normalize( drho );
				v = normalize( v - dot( v, drho ) * drho );
			}
#else
			v = normalize(v);
#endif
			force_x[i] += v.x * bodyforce;
			force_y[i] += v.y * bodyforce;
			force_z[i] += v.z * bodyforce;
		}
    }
}

void FixEggForce::post_force(int evflag)
{
	MesoNeighList *list = meso_neighbor->lists_device[ force->pair->list->index ];
    if( !list ) error->all( FLERR, "<MESO> fix aboundary/meso must be used together with a pair from USER-MESO" );

    meso_atom->meso_avec->dp2sp_merged( 0, 0, atom->nlocal+atom->nghost, true );
	meso_atom->tex_misc("mask").bind( *(meso_atom->dev_mask) );

    static GridConfig grid_cfg;
    if (!grid_cfg.x) {
    	grid_cfg = meso_device->configure_kernel( gpu_eggforce );
    }

	gpu_eggforce<<< grid_cfg.x, grid_cfg.y, 0, meso_device->stream() >>> (
		meso_atom->tex_coord_merged,
		meso_atom->tex_veloc_merged,
		meso_atom->tex_misc("mask"),
		meso_atom->tex_tag,
		meso_atom->dev_force(0), meso_atom->dev_force(1), meso_atom->dev_force(2),
        list->dev_pair_count_core,
        list->dev_pair_table,
        rho0,
        bodyforce,
        1 / sqrt( pow( 2*M_PI, 3. ) * pow( sigma, 3. ) ),
        sigma,
        list->n_col,
        groupbit,
        ref_groupbit,
        atom->nlocal );
}
