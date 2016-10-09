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
#include "fix_rdf_meso.h"
#include "group.h"
#include "neighbor_meso.h"
#include "neigh_list_meso.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

MesoFixRDF::MesoFixRDF( LAMMPS *lmp, int narg, char **arg ):
    Fix( lmp, narg, arg ),
    MesoPointers( lmp ),
    dev_histogram( lmp, "MesoFixRDF::dev_histogram" ),
    n_hist(0),
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
            n_hist = atoi( arg[i] );
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

    if( output == "" || n_hist == 0 ) {
        error->all( FLERR, "Incomplete compute rdf command: insufficient arguments" );
    }

    dev_histogram.grow( n_hist, false, true );

    if( !force->pair ) error->all( FLERR, "<MESO> fix rho/meso must be used together with a pair style" );
}

MesoFixRDF::~MesoFixRDF()
{
    dump();
}

void MesoFixRDF::init()
{
    rc = force->pair->cutforce;
}

int MesoFixRDF::setmask()
{
    int mask = 0;
    mask |= FixConst::POST_FORCE;
    return mask;
}

void MesoFixRDF::setup( int vflag )
{
    post_force(0);
}

template<int SLOT_PER_WARP, int NWORD_PER_SLOT>
__global__ void gpu_compute_rdf(
	uint* __restrict histogram,
	texobj tex_aid,
    texobj tex_coord_merged,
    const int* __restrict mask,
    const int* __restrict bin_head,
    const int* __restrict bin_size,
    const int* __restrict stencil_len,
    const int* __restrict stencil,
    const int stencil_padding,
    const r32 rc,
    const r32 hist_sz_inv,
    const int n_hist,
    const int groupi,
    const int groupj,
    const int n_cell
)
{
    if( SLOT_PER_WARP > WARPSZ )  // no register overhead because can be eliminated entirely if predicate is static
        printf( "<MESO> slot number %d exceeds warp size %d\n", SLOT_PER_WARP, WARPSZ );
    extern __shared__ int SMEM[];

    uint  *hist_local = (uint*) SMEM;
    r32   *x     = ( r32* )   &SMEM[ n_hist + __warpid_local() * SLOT_PER_WARP * NWORD_PER_SLOT + 0 * SLOT_PER_WARP ];
    r32   *y     = ( r32* )   &SMEM[ n_hist + __warpid_local() * SLOT_PER_WARP * NWORD_PER_SLOT + 1 * SLOT_PER_WARP ];
    r32   *z     = ( r32* )   &SMEM[ n_hist + __warpid_local() * SLOT_PER_WARP * NWORD_PER_SLOT + 2 * SLOT_PER_WARP ];
    int   *aid   = ( int* )   &SMEM[ n_hist + __warpid_local() * SLOT_PER_WARP * NWORD_PER_SLOT + 3 * SLOT_PER_WARP ];

    for( int i = threadIdx.x; i < n_hist ; i += blockDim.x ) hist_local[i] = 0;
    __syncthreads();

    const int  lane_id = __laneid();

    for( int bin_id = __warpid_global(); bin_id < n_cell; bin_id += __warp_num_global() ) {
        int cur_bin_size  = bin_size[ bin_id ] ;
        if( !cur_bin_size ) continue;

        for( int p = 0 ; p < cur_bin_size ; p += SLOT_PER_WARP ) {
            // load shared data
            int pack = min( cur_bin_size - p, SLOT_PER_WARP );
            int idx = tex1Dfetch<int>( tex_aid, bin_head[ bin_id ] + p + lane_id );
            bool masked = false;
            uint vmasked;
            if( lane_id < pack ) masked = mask[ idx ] & groupi;
			vmasked = __ballot( masked );
			if ( masked ) {
				float4 v     = tex1Dfetch<float4>( tex_coord_merged, idx );
				int p_ins = __popc( vmasked & __lanemask_lt() );
				x[p_ins] = v.x;
				y[p_ins] = v.y;
				z[p_ins] = v.z;
				aid[p_ins] = idx;
			}
			pack = __popc( vmasked );

            // load batch of atoms from stencil and compare against center atom
            for( int j = lane_id ; j < stencil_len[ bin_id ] ; j += warpSize ) {
                int aid_j = stencil[ bin_id * stencil_padding + j ];

                if ( mask[aid_j] & groupj ) {
					r32  xj, yj, zj;
					{
						float4 v = tex1Dfetch<float4>( tex_coord_merged, aid_j );
						xj = v.x, yj = v.y, zj = v.z;
					}

					// examine stencil atoms against each atom in my bin
					for( int i = 0 ; i < pack ; i++ ) {
						r32 dx    = x[i] - xj ;
						r32 dy    = y[i] - yj ;
						r32 dz    = z[i] - zj ;
						r32 dr2   = dx * dx + dy * dy + dz * dz ;

						if ( dr2 < rc * rc && aid[i] != aid_j ) {
							r32 r = dr2 * rsqrtf(dr2);
							int bid = floorf( r * hist_sz_inv );
							atomicInc( hist_local + bid, 0xFFFFFFFFU );
						}
					}
                }
            }
        }
    }

	__syncthreads();

	for( int i = threadIdx.x; i < n_hist ; i += blockDim.x ) atomic_add( histogram+i, hist_local[i] );
}

void MesoFixRDF::post_force(int evflag)
{
    const static int nslot = 32;
    const static int nword = 4;

    static GridConfig grid_cfg;
    if( !grid_cfg.x ) {
        grid_cfg = meso_device->occu_calc.right_peak( 0, gpu_compute_rdf<nslot, nword>, 0, cudaFuncCachePreferShared );
        cudaFuncSetCacheConfig( gpu_compute_rdf<nslot, nword>, cudaFuncCachePreferShared );
    }

    if ( update->ntimestep % n_every == 0 ) {

		MesoNeighList *dlist = meso_neighbor->lists_device[ force->pair->list->index ];

        int n_cell       = meso_neighbor->mbins();
        int shmem_size = dev_histogram.n_byte() + ( grid_cfg.y / WARPSZ ) * ( nslot * nword * sizeof( int ) );

         gpu_compute_rdf<nslot, nword> <<< grid_cfg.x, grid_cfg.y, shmem_size, meso_device->stream() >>> (
			dev_histogram,
			meso_neighbor->cuda_bin.tex_atm_id,
    		meso_atom->tex_coord_merged,
    		meso_atom->dev_mask,
    		meso_neighbor->cuda_bin.dev_bin_location,
    		meso_neighbor->cuda_bin.dev_bin_size_local,
    		dlist->dev_stencil_len,
    		dlist->dev_stencil,
    		dlist->dev_stencil.pitch_elem(),
    		rc,
    		n_hist/rc,
    		n_hist,
    		groupbit,
    		j_groupbit,
    		n_cell );

		n_steps++;
    }
}

void MesoFixRDF::dump()
{
    // dump result
	std::vector<uint> histogram( n_hist );
    dev_histogram.download( &histogram[0], dev_histogram.n_elem(), meso_device->stream() );
    meso_device->sync_device();

    long long n[2] = {0, 0};
    for(int i = 0 ; i < atom->nlocal ; i++) {
    	if ( atom->mask[i] &   groupbit ) ++n[0];
    	if ( atom->mask[i] & j_groupbit ) ++n[1];
    }
    MPI_Reduce( comm->me == 0 ? MPI_IN_PLACE : n, n, 2, MPI_LONG_LONG, MPI_SUM, 0, world );

    std::vector<uint> histogram_master( n_hist, 0 );
    MPI_Reduce( histogram.data(), histogram_master.data(), n_hist, MPI_UNSIGNED, MPI_SUM, 0, MPI_COMM_WORLD );

    double V = 1.0;
    for(int d = 0 ; d < 3 ; d++) V *= domain->boxhi[d] - domain->boxlo[d];

    double rho = n[1] / V;

    if( comm->me == 0 ) {
        std::ofstream fout;
        fout.open( output.c_str() );
        fout << std::setprecision( 15 );
        double bin_sz = rc / n_hist;
        for( int i = 0 ; i < n_hist ; i++ ) {
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

