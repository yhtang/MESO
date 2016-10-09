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

#include "atom_meso.h"
#include "atom_vec_meso.h"
#include "comm_meso.h"
#include "engine_meso.h"
#include "fix_rdf_fast_meso.h"
#include "group.h"
#include "neighbor_meso.h"
#include "neigh_list_meso.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

MesoFixRDFFast::MesoFixRDFFast( LAMMPS *lmp, int narg, char **arg ):
    Fix( lmp, narg, arg ),
    MesoPointers( lmp ),
    dev_histogram( lmp, "MesoFixRDFFast::dev_histogram" ),
    n_bin(0),
    n_steps(0),
    n_every(1)
{
	j_groupbit = groupbit;

    for( int i = 0 ; i < narg ; i++ ) {
        if( !strcmp( arg[i], "output" ) ) {
            if( ++i >= narg ) error->all( FLERR, "Incomplete compute vprof command after 'output'" );
            output = arg[i];
        } else if( !strcmp( arg[i], "nbin" ) ) {
            if( ++i >= narg ) error->all( FLERR, "Incomplete compute vprof command after 'nbin'" );
            n_bin = atoi( arg[i] );
		} else if( !strcmp( arg[i], "every" ) ) {
			if( ++i >= narg ) error->all( FLERR, "Incomplete compute vprof command after 'every'" );
			n_every = atoi( arg[i] );
		} else if( !strcmp( arg[i], "other" ) ) {
			if( ++i >= narg ) error->all( FLERR, "Incomplete compute vprof command after 'other'" );
		    if ( ( j_group = group->find( arg[i] ) ) == -1 ) {
		    	error->all( FLERR, "<MESO> Undefined other group id in fix rdf/meso" );
		    }
		    j_groupbit = group->bitmask[ j_group ];
		}
    }

    if( output == "" || n_bin == 0 ) {
        error->all( FLERR, "Incomplete compute rdf command: insufficient arguments" );
    }

    dev_histogram.grow( n_bin, false, true );

    if( !force->pair ) error->all( FLERR, "<MESO> fix rho/meso must be used together with a pair style" );
}

MesoFixRDFFast::~MesoFixRDFFast()
{
    dump();
}

void MesoFixRDFFast::init()
{
    rc = force->pair->cutforce;
}

int MesoFixRDFFast::setmask()
{
    int mask = 0;
    mask |= FixConst::POST_FORCE;
    return mask;
}

void MesoFixRDFFast::setup( int vflag )
{
}

__global__ void gpu_calc_rdf(
    texobj tex_coord,
    int* __restrict mask,
    int* __restrict pair_count,
    int* __restrict pair_table,
    uint* __restrict histogram,
    const int pair_padding,
    const r32 rc,
    const r32 bin_sz_inv,
    const int nbin,
    const int groupi,
    const int groupj,
    const int n )
{
    extern __shared__ uint hist_local[];
    for( int i = threadIdx.x; i < nbin ; i += blockDim.x ) hist_local[i] = 0;
    __syncthreads();

	for( int i = threadIdx.x + blockIdx.x * blockDim.x; i < n ; i += gridDim.x * blockDim.x ) {
		if ( !( mask[i] & groupi ) ) continue;

		f3u  coord1 = tex1Dfetch<float4>( tex_coord, i );
		int  n_pair = pair_count[i];
		int *p_pair = pair_table + ( i - __laneid() ) * pair_padding + __laneid();

		for( int p = 0; p < n_pair; p++ ) {
			int j   = __lds( p_pair );
			p_pair += pair_padding;
			if( ( p & 31 ) >= WARPSZ - 1 ) p_pair -= WARPSZ * pair_padding - WARPSZ;
			if ( !( mask[j] & groupj ) ) continue;

			f3u coord2   = tex1Dfetch<float4>( tex_coord, j );
			r32 dx       = coord1.x - coord2.x;
			r32 dy       = coord1.y - coord2.y;
			r32 dz       = coord1.z - coord2.z;
			r32 rsq      = dx * dx + dy * dy + dz * dz;

			if( rsq < rc * rc ) {
				r32 r = rsq * rsqrtf(rsq);
				int bid = floorf( r * bin_sz_inv );
				atomicInc( hist_local + bid, 0xFFFFFFFFU );
			}
		}
    }
	__syncthreads();

	for( int i = threadIdx.x; i < nbin ; i += blockDim.x ) atomic_add( histogram+i, hist_local[i] );
}

void MesoFixRDFFast::post_force(int evflag)
{
    static GridConfig grid_cfg;
    if( !grid_cfg.x ) {
        grid_cfg = meso_device->occu_calc.right_peak( 0, gpu_calc_rdf, 0, cudaFuncCachePreferShared );
        cudaFuncSetCacheConfig( gpu_calc_rdf, cudaFuncCachePreferShared );
    }

    if ( update->ntimestep % n_every == 0 ) {

		MesoNeighList *dlist = meso_neighbor->lists_device[ force->pair->list->index ];

		gpu_calc_rdf<<< grid_cfg.x, grid_cfg.y, dev_histogram.n_byte(), meso_device->stream() >>> (
			meso_atom->tex_coord_merged,
			meso_atom->dev_mask,
			dlist->dev_pair_count_core,
			dlist->dev_pair_table,
			dev_histogram,
			dlist->n_col,
			rc,
			n_bin/rc,
			n_bin,
			groupbit,
			j_groupbit,
			atom->nlocal );

		n_steps++;
    }
}

void MesoFixRDFFast::dump()
{
    // dump result
	std::vector<uint> histogram( n_bin );
    dev_histogram.download( &histogram[0], dev_histogram.n_elem(), meso_device->stream() );
    meso_device->sync_device();

    long long n[2] = {0, 0};
    for(int i = 0 ; i < atom->nlocal ; i++) {
    	if ( atom->mask[i] &   groupbit ) ++n[0];
    	if ( atom->mask[i] & j_groupbit ) ++n[1];
    }
    MPI_Reduce( comm->me == 0 ? MPI_IN_PLACE : n, n, 2, MPI_LONG_LONG, MPI_SUM, 0, world );

    std::vector<uint> histogram_master( n_bin, 0 );
    MPI_Reduce( histogram.data(), histogram_master.data(), n_bin, MPI_UNSIGNED, MPI_SUM, 0, MPI_COMM_WORLD );

    double V = 1.0;
    for(int d = 0 ; d < 3 ; d++) V *= domain->boxhi[d] - domain->boxlo[d];
    double rho = n[1] / V;

    if( comm->me == 0 ) {
        std::ofstream fout;
        fout.open( output.c_str() );
        fout << std::setprecision( 15 );
        double bin_sz = rc / n_bin;
        for( int i = 0 ; i < n_bin ; i++ ) {
        	double freq = histogram_master[i] / double(n[0]) / double(n_steps);
        	double gr;
        	if ( i == 0 ) gr = freq / ( 4./3. * 3.1415 * std::pow(bin_sz,3.) ) / rho;
        	else gr = freq / ( 4./3. * 3.1415 * ( std::pow(bin_sz*(i+1),3.) - std::pow(bin_sz*i,3.)  ) ) / rho;
        	fout<< ( i + 0.5 ) * bin_sz << '\t'
        	    << gr << '\t'
        	    << std::endl;
        }
        fout.close();
    }

    MPI_Barrier( world );
}

