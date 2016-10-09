#include "mpi.h"
#include "string.h"
#include "update.h"
#include "force.h"
#include "domain.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "modify.h"
#include "fix.h"
#include "group.h"
#include "error.h"

#include "atom_meso.h"
#include "atom_vec_meso.h"
#include "neighbor_meso.h"
#include "neigh_list_meso.h"
#include "compute_ec_edpd_meso.h"
#include "pair_edpd_trp_base_meso.h"
#include "engine_meso.h"

using namespace LAMMPS_NS;
using namespace PNIPAM_COEFFICIENTS;

MesoComputeEcEDPD::MesoComputeEcEDPD( LAMMPS *lmp, int narg, char **arg ) :
    Compute( lmp, narg, arg ),
    MesoPointers( lmp ),
    dev_energy( lmp, "MesoComputeEcEDPD::dev_energy" ),
    dev_ninter( lmp, "MesoComputeEcEDPD::dev_ninter" ),
    hst_energy( lmp, "MesoComputeEcEDPD::hst_energy" ),
    hst_ninter( lmp, "MesoComputeEcEDPD::hst_ninter" ),
	jgroup(0), jgroupbit(0),
	per_pair(true),
	pair(NULL)
{
    if( narg < 4 ) error->all( __FILE__, __LINE__, "<MESO> Compute interaction/edpd/meso usage: id style igroup jgroup [per_pair]" );

    if ( ( jgroup = group->find( arg[3] ) ) == -1 ) {
    	error->all( FLERR, "<MESO> Undefined group id in compute interaction/edpd/meso" );
    }
    jgroupbit = group->bitmask[ jgroup ];

    if ( narg > 4 ) {
    	if ( !strcmp( arg[4], "true" ) || !strcmp( arg[4], "yes" ) ) per_pair = true;
    	else if ( !strcmp( arg[4], "false" ) || !strcmp( arg[4], "no" ) ) per_pair = false;
    	else per_pair = atoi( arg[4] );
    }

    scalar_flag = 1;
    vector_flag = 0;
    extscalar = 0;

    hst_energy.grow(1);
    hst_ninter.grow(1);

    pair = dynamic_cast<MesoPairEDPDTRPBase*>( force->pair );
    if( !pair ) error->all( FLERR, "<MESO> compute interaction/edpd/meso must be used together with pair edpd/pnipam/meso" );
}

MesoComputeEcEDPD::~MesoComputeEcEDPD()
{
}

void MesoComputeEcEDPD::setup()
{
}

__global__ void gpu_potential_ec_trp(
    texobj tex_coord, texobj tex_therm,
    const int* __restrict mask,
    int* __restrict pair_count, int* __restrict pair_table,
    r64* __restrict e_pair,
    int* __restrict n_inter,
    r64* __restrict coefficients,
    const r64 dt_inv_sqrt,
    const int pair_padding,
    const int n_type,
    const int igroupbit,
    const int jgroupbit,
    const int n_all
)
{
    extern __shared__ r64 coeffs[];
    for( int p = threadIdx.x; p < n_type * n_type * n_coeff; p += blockDim.x )
        coeffs[p] = coefficients[p];
    __syncthreads();

    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_all; i += gridDim.x * blockDim.x ) {
		r64  energy = 0.;
		int  ninter = 0;

		if ( (mask[i]&igroupbit) || (mask[i]&jgroupbit) ) {
			f3u  coord1 = tex1Dfetch<float4>( tex_coord, i );
			tmm  therm1 = tex1Dfetch<float4>( tex_therm,  i );
			int  imask  = mask[i];

			int  n_pair = pair_count[i];
			int *p_pair = pair_table + ( i - __laneid() ) * pair_padding + __laneid();

			for( int p = 0; p < n_pair; p++ ) {
				int j   = __lds( p_pair );
				p_pair += pair_padding;
				if( ( p & 31 ) == 31 ) p_pair -= 32 * pair_padding - 32;
				if ( ! ( ( (imask&igroupbit)&&(mask[j]&jgroupbit) ) || ( (imask&jgroupbit)&&(mask[j]&igroupbit) ) ) ) continue;

				f3u coord2   = tex1Dfetch<float4>( tex_coord, j );
				r64 dx       = coord1.x - coord2.x;
				r64 dy       = coord1.y - coord2.y;
				r64 dz       = coord1.z - coord2.z;
				r64 rsq      = dx * dx + dy * dy + dz * dz;
				r64 *coeff_ij = coeffs + ( coord1.i * n_type + coord2.i ) * n_coeff;

				if( rsq < power<2>( coeff_ij[p_cut] ) && rsq >= EPSILON_SQ ) {
					r64 rinv     = rsqrt( rsq );
					r64 r        = rsq * rinv;

					tmm therm2   = tex1Dfetch<float4>( tex_therm, j );
					r64 T_ij     = 0.5 * ( therm1.T + therm2.T );

					r64 alpha_ij = coeff_ij[p_a0] * T_ij; // DPD classical term
					r64 d_alpha  = coeff_ij[p_da] / ( 1.0 + expf( coeff_ij[p_omega] * ( T_ij - coeff_ij[p_theta]  ) ) ); // temperature-dependence
					r64 wc       = 1.0 - r * coeff_ij[p_cutinv];
					energy      += 0.5 * wc * wc * (alpha_ij + d_alpha);
					ninter++;
				}
			}
    	}

		// each pair computed twice now
		e_pair[i] = energy;
		n_inter[i] = ninter;
    }
}

double MesoComputeEcEDPD::compute_scalar()
{
    invoked_scalar = update->ntimestep;

    pair->prepare_coeff();
    meso_atom->meso_avec->dp2sp_merged( 0, 0, atom->nlocal+atom->nghost, true );
    if (dev_energy.n_elem() < atom->nlocal) {
    	dev_energy.grow( atom->nlocal );
    	dev_ninter.grow( atom->nlocal );
    }
    MesoNeighList *dlist = meso_neighbor->lists_device[ pair->list->index ];

    static GridConfig grid_cfg = meso_device->configure_kernel( gpu_potential_ec_trp, pair->dev_coefficients.n_byte() );
    gpu_potential_ec_trp<<< grid_cfg.x, grid_cfg.y, pair->dev_coefficients.n_byte(), meso_device->stream() >>> (
		meso_atom->tex_coord_merged,
		meso_atom->tex_misc("therm"),
		meso_atom->dev_mask,
        dlist->dev_pair_count_core,
        dlist->dev_pair_table,
        dev_energy,
        dev_ninter,
        pair->dev_coefficients,
        1.0 / sqrt( update->dt ),
        dlist->n_col,
        atom->ntypes,
        groupbit,
        jgroupbit,
        atom->nlocal );

    size_t threads_per_block = meso_device->query_block_size( gpu_reduce_sum_host<double> );
    gpu_reduce_sum_host <<< 1, threads_per_block, 0, meso_device->stream() >>> (
        dev_energy.ptr(), hst_energy.ptr(), atom->nlocal );
    gpu_reduce_sum_host <<< 1, threads_per_block, 0, meso_device->stream() >>> (
        dev_ninter.ptr(), hst_ninter.ptr(), atom->nlocal );
    meso_device->stream().sync();

    double mine[2], total[2];
    mine[0] = hst_energy[0];
    mine[1] = hst_ninter[0];
    MPI_Allreduce( mine, total, 2, MPI_DOUBLE, MPI_SUM, world );
    if (per_pair) return ( scalar = total[0] / total[1] );
    else return ( scalar = total[0] );
}

