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
#include "pair_edpd_trp_base_meso.h"
#include "fix_edpd_energy_convert_meso.h"

using namespace LAMMPS_NS;
using namespace PNIPAM_COEFFICIENTS;

FixEDPDEnergyConvert::FixEDPDEnergyConvert( LAMMPS *lmp, int narg, char **arg ) :
    Fix( lmp, narg, arg ),
    MesoPointers( lmp ),
    dev_dE  ( lmp, "FixEDPDEnergyConvert::dev_dE" ),
    dev_Q_in( lmp, "FixEDPDEnergyConvert::dev_Q_in" )
{
    if( narg != 4 ) error->all( __FILE__, __LINE__, "<MESO> Compute ecvt/edpd/meso usage: id igroup style jgroup" );

    if ( ( jgroup = group->find( arg[3] ) ) == -1 ) {
    	error->all( FLERR, "<MESO> Undefined group id in compute interaction/edpd/meso" );
    }
    jgroupbit = group->bitmask[ jgroup ];

    pair = dynamic_cast<MesoPairEDPDTRPBase*>( force->pair );
    if( !pair ) error->all( FLERR, "<MESO> fix ecvt/edpd/meso must be used together with pair edpd/pnipam/meso" );

    dev_dE.grow(2);
}

int FixEDPDEnergyConvert::setmask()
{
    int mask = 0;
    mask |= FixConst::INITIAL_INTEGRATE;
    return mask;
}


__global__ void gpu_convert_energy(
    texobj tex_coord, texobj tex_therm,
    const r64* __restrict Q_in,
    const int* __restrict mask,
    int* __restrict pair_count,
    int* __restrict pair_table,
    r64* __restrict coefficients,
    r64* __restrict Q_out,
    r32* dE,
    const r64 dt,
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
    	if ( !(mask[i]&igroupbit) && !(mask[i]&jgroupbit) ) continue;

		f3u  coord1 = tex1Dfetch<float4>( tex_coord, i );
		tmm  therm1 = tex1Dfetch<float4>( tex_therm,  i );
		int  imask  = mask[i];
		r64 Ti_new   = therm1.T + Q_in[i]*dt / coeffs[(coord1.i*n_type+coord1.i)*n_coeff+p_cv];

		int  n_pair = pair_count[i];
		int *p_pair = pair_table + ( i - __laneid() ) * pair_padding + __laneid();
		r64  dQ = 0.;

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
				r64 d_alpha  = coeff_ij[p_da] / ( 1.0 + expf( coeff_ij[p_omega] * ( T_ij - coeff_ij[p_theta]  ) ) ); // temperature-dependence
				r64 Tj_new   = therm2.T + Q_in[j]*dt / coeff_ij[p_cv];
				r64 T_ij_new = 0.5 * ( Ti_new + Tj_new );
				r64 d_alpha_new = coeff_ij[p_da] / ( 1.0 + expf( coeff_ij[p_omega] * ( T_ij_new - coeff_ij[p_theta]  ) ) ); // temperature-dependence
				r64 dd_alpha = d_alpha_new - d_alpha;
				r64 wc       = 1.0 - r * coeff_ij[p_cutinv];
				r64 dq       = 0.5 * wc * wc * dd_alpha / dt;
				dQ          -= dq;
			}
		}

		Q_out[i] += dQ * 0.5;
		atomic_add( dE+0, r32(dQ) );
		atomic_add( dE+1, 1.0f  );
	}
}

__global__ void print_e(r32 *e) {
	printf("e_total: %f, e/%f = %f\n", e[0], e[1], e[0]/e[1] );
}

void FixEDPDEnergyConvert::initial_integrate(int evflag)
{
    pair->prepare_coeff();
    meso_atom->meso_avec->dp2sp_merged( 0, 0, atom->nlocal+atom->nghost, true );
    MesoNeighList *dlist = meso_neighbor->lists_device[ pair->list->index ];

    std::vector<CUDAEvent> events;
    CUDAEvent prev_work = meso_device->event( "FixSPHRhoMeso::gpu_compute_rho" );
    prev_work.record( meso_device->stream() );
    CUDAStream::all_waiton( prev_work );
    events = meso_atom->meso_avec->transfer( AtomAttribute::BORDER | AtomAttribute::HEAT, CUDACOPY_G2C );
    events.back().sync();

    meso_comm->forward_comm_fix( this );

    events = meso_atom->meso_avec->transfer( AtomAttribute::GHOST | AtomAttribute::HEAT, CUDACOPY_C2G );
    meso_device->stream().waiton( events.back() );

    dev_dE.set( 0., meso_device->stream() );
    if (dev_Q_in.n_elem()<(*meso_atom->dev_Q).n_elem()) dev_Q_in.grow( (*meso_atom->dev_Q).n_elem() );
    dev_Q_in.upload( meso_atom->dev_Q, (*meso_atom->dev_Q).n_elem(), meso_device->stream() );

    static GridConfig grid_cfg = meso_device->configure_kernel( gpu_convert_energy, pair->dev_coefficients.n_byte() );
    gpu_convert_energy<<< grid_cfg.x, grid_cfg.y, pair->dev_coefficients.n_byte(), meso_device->stream() >>> (
		meso_atom->tex_coord_merged,
		meso_atom->tex_misc("therm"),
		dev_Q_in,
		meso_atom->dev_mask,
        dlist->dev_pair_count_core,
        dlist->dev_pair_table,
        pair->dev_coefficients,
		meso_atom->dev_Q,
		dev_dE,
        update->dt,
        dlist->n_col,
        atom->ntypes,
        groupbit,
        jgroupbit,
        atom->nlocal );

//	print_e<<<1,1,0,meso_device->stream()>>>(e);
}

int FixEDPDEnergyConvert::pack_comm( int n, int *list, double *buf, int pbc_flag, int *pbc )
{
    int m = 0;
    for( int i = 0; i < n; i++ ) {
        buf[m++] = atom->Q[ list[i] ];
    }
    return 1;
}

void FixEDPDEnergyConvert::unpack_comm( int n, int first, double *buf )
{
    int m = 0;
    int last = first + n;
    for( int i = first; i < last; i++ ) {
        atom->Q[i] = buf[m++];
    }
}


