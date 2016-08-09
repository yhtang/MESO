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

#include "atom_meso.h"
#include "atom_vec_meso.h"
#include "comm_meso.h"
#include "engine_meso.h"
#include "fix_sph_rho_meso.h"
#include "neighbor_meso.h"
#include "neigh_list_meso.h"
#include "pair_sph_meso.h"

using namespace LAMMPS_NS;
using namespace SPH_TWOBODY;

FixSPHRhoMeso::FixSPHRhoMeso( LAMMPS *lmp, int narg, char **arg ) :
    Fix( lmp, narg, arg ),
    MesoPointers( lmp )
{
    pair = dynamic_cast<MesoPairSPH*>( force->pair );
    if( !pair ) error->all( FLERR, "<MESO> fix rho/meso must be used together with pair sph/meso" );
}

int FixSPHRhoMeso::setmask()
{
    int mask = 0;
    mask |= FixConst::PRE_FORCE;
    return mask;
}

__global__ void gpu_compute_rho(
    texobj tex_coord,
    texobj tex_mass,
    r64* __restrict rho,
    int* __restrict pair_count,
    int* __restrict pair_table,
    r64* __restrict coefficients,
    const int pair_padding,
    const int n_type,
    const int p_beg,
    const int p_end,
    const int n_part )
{
    int block_per_part = gridDim.x / n_part;
    int part_id = blockIdx.x / block_per_part;
    if( part_id >= n_part ) return;
    int part_size = block_per_part * blockDim.x;
    int id_in_partition = blockIdx.x % block_per_part * blockDim.x + threadIdx.x;

    extern __shared__ r64 coeffs[];
    for( int p = threadIdx.x; p < n_type * n_type * n_coeff2; p += blockDim.x )
        coeffs[p] = coefficients[p];
    __syncthreads();

    for( int iter = id_in_partition; ; iter += part_size ) {
        int i = ( p_beg & WARPALIGN ) + iter;
        if( i >= p_end ) break;
        if( i >= p_beg ) {
            f3u  coord1 = tex1Dfetch<float4>( tex_coord, i );
            int  n_pair = pair_count[i];
            int *p_pair = pair_table + ( i - __laneid() + part_id ) * pair_padding + __laneid();
            r64 *coeff_ii = coeffs + ( coord1.i * n_type + coord1.i ) * n_coeff2;
            r64  rhosum = 0.0;
            if( part_id == 0 ) {
//              SPHKernelGauss3D<r64> kernel_self(coeff_ii[p_cut]);
//              r64 W = kernel_self(0.0);
                SPHKernelTang3D<r64> kernel( coeff_ii[p_cutinv] );
                r64 W = kernel( r64( 0 ) ) * coeff_ii[p_n3];
                rhosum += tex1Dfetch<r64>( tex_mass, i ) * W;
            }

            for( int p = part_id; p < n_pair; p += n_part ) {
                int j   = __lds( p_pair );
                p_pair += pair_padding * n_part;
                if( ( p & 31 ) + n_part >= WARPSZ ) p_pair -= WARPSZ * pair_padding - WARPSZ;

                f3u coord2   = tex1Dfetch<float4>( tex_coord, j );
                r64 dx       = coord1.x - coord2.x;
                r64 dy       = coord1.y - coord2.y;
                r64 dz       = coord1.z - coord2.z;
                r64 rsq      = dx * dx + dy * dy + dz * dz;
                r64 *coeff_ij = coeffs + ( coord1.i * n_type + coord2.i ) * n_coeff2;

                if( rsq < power<2>( coeff_ij[p_cut] ) && rsq >= EPSILON_SQ ) {
                    r64 rinv = rsqrt( rsq );
                    r64 r = rsq * rinv;
//                  SPHKernelGauss3D<r64> kernel(coeff_ij[p_cut]);
//                  r64 W = kernel( r );
                    SPHKernelTang3D<r64> kernel( coeff_ij[p_cutinv] );
                    r64 W = kernel( r ) * coeff_ij[p_n3];
                    rhosum += tex1Dfetch<r64>( tex_mass, j ) * W;
                }
            }

            if( n_part == 1 ) {
                rho[i] = rhosum;
            } else {
                atomic_add( rho + i, rhosum );
            }
        }
    }
}

__global__ void check_rho( r64 *rho, int n )
{
    for( int i = 0; i < n; i++ ) printf( "%.4lf\n", rho[i] );
}

void FixSPHRhoMeso::pre_force( __attribute__( ( unused ) ) int vflag )
{
    prepare_coeff();

    MesoNeighList *dlist = meso_neighbor->lists_device[ pair->list->index ];

    meso_atom->meso_avec->dp2sp_merged( 0, 0, atom->nlocal, true );

    int shared_mem_size = atom->ntypes * atom->ntypes * n_coeff2 * sizeof( r64 );
    static GridConfig grid_cfg = meso_device->configure_kernel( gpu_compute_rho, shared_mem_size );
    gpu_compute_rho <<< grid_cfg.x, grid_cfg.y, shared_mem_size, meso_device->stream() >>> (
        meso_atom->tex_coord_merged,
        meso_atom->tex_mass,
        meso_atom->dev_rho,
        dlist->dev_pair_count_core, dlist->dev_pair_table,
        pair->dev_coeffs2,
        dlist->n_col, atom->ntypes,
        0, atom->nlocal,
        grid_cfg.partition( atom->nlocal, WARPSZ )
    );

    std::vector<CUDAEvent> events;
    CUDAEvent prev_work = meso_device->event( "FixSPHRhoMeso::gpu_compute_rho" );
    prev_work.record( meso_device->stream() );
    CUDAStream::all_waiton( prev_work );
    events = meso_atom->meso_avec->transfer( AtomAttribute::BORDER | AtomAttribute::RHO, CUDACOPY_G2C );
    events.back().sync();

    meso_comm->forward_comm_fix( this );

    //  there is no need to sync with any previous GPU events
    //  because this transfer only depends on data on the CPU
    //  which is surly to be ready when post_comm is called
    events = meso_atom->meso_avec->transfer( AtomAttribute::GHOST | AtomAttribute::RHO, CUDACOPY_C2G );
    meso_device->stream().waiton( events.back() );

//  check_rho<<<1,1>>>(meso_atom->dev_rho,atom->nlocal+atom->nghost);
//  fast_exit(1);
}

void FixSPHRhoMeso::setup_pre_force( int vflag )
{
    pre_force( vflag );
}

void FixSPHRhoMeso::prepare_coeff()
{
    pair->prepare_coeff();
}

int FixSPHRhoMeso::pack_comm( int n, int *list, double *buf, int pbc_flag, int *pbc )
{
    int m = 0;
    for( int i = 0; i < n; i++ ) {
        buf[m++] = atom->rho[ list[i] ];
    }
    return 1;
}

void FixSPHRhoMeso::unpack_comm( int n, int first, double *buf )
{
    int m = 0;
    int last = first + n;
    for( int i = first; i < last; i++ ) {
        atom->rho[i] = buf[m++];
    }
}


