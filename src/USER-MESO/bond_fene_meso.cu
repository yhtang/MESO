#include "mpi.h"
#include "math.h"
#include "stdlib.h"
#include "domain.h"
#include "force.h"
#include "memory.h"
#include "error.h"
#include "update.h"

#include "engine_meso.h"
#include "comm_meso.h"
#include "atom_meso.h"
#include "atom_vec_meso.h"
#include "bond_fene_meso.h"
#include "neighbor_meso.h"

using namespace LAMMPS_NS;

//texture<float4, cudaTextureType1D, cudaReadModeElementType> tex_coord_hb;

MesoBondFENE::MesoBondFENE( LAMMPS *lmp ):
    BondFENE( lmp ),
    MesoPointers( lmp ),
    dev_k( lmp, "MesoBondFENE::dev_k" ),
    dev_r0( lmp, "MesoBondFENE::dev_r0" ),
    dev_epsilon( lmp, "MesoBondFENE::dev_epsilon" ),
    dev_sigma( lmp, "MesoBondFENE::dev_sigma" )
{
    coeff_alloced = 0;
}

MesoBondFENE::~MesoBondFENE()
{
}

void MesoBondFENE::alloc_coeff()
{
    if( coeff_alloced ) return;

    coeff_alloced = 1;
    int n = atom->nbondtypes;
    dev_k      .grow( n + 1, false, false );
    dev_r0     .grow( n + 1, false, false );
    dev_epsilon.grow( n + 1, false, false );
    dev_sigma  .grow( n + 1, false, false );
    dev_k      .upload( k,       n + 1, meso_device->stream() );
    dev_r0     .upload( r0,      n + 1, meso_device->stream() );
    dev_epsilon.upload( epsilon, n + 1, meso_device->stream() );
    dev_sigma  .upload( sigma,   n + 1, meso_device->stream() );
}

#define TWO_1_3 1.25992104989487316477

template<int evflag>
__global__ void gpu_bond_fene(
    texobj tex_coord_merged,
    r64*  __restrict force_x,
    r64*  __restrict force_y,
    r64*  __restrict force_z,
    r64*  __restrict virial_xx,
    r64*  __restrict virial_yy,
    r64*  __restrict virial_zz,
    r64*  __restrict virial_xy,
    r64*  __restrict virial_xz,
    r64*  __restrict virial_yz,
    r64*  __restrict e_bond,
    int*  __restrict nbond,
    int2* __restrict bonds,
    r64*  __restrict k_global,
    r64*  __restrict r0_global,
    r64*  __restrict epsilon_global,
    r64*  __restrict sigma_global,
    const double3 period,
    const int padding,
    const int n_type,
    const int n_local )
{
    extern __shared__ r64 shared_data[];
    r64 *k       = &shared_data[0];
    r64 *r0      = k       + n_type + 1;
    r64 *epsilon = r0      + n_type + 1;
    r64 *sigma   = epsilon + n_type + 1;
    for( int i = threadIdx.x ; i < n_type + 1 ; i += blockDim.x ) {
        k[i]       = k_global [i];
        r0[i]      = r0_global[i];
        epsilon[i] = epsilon_global[i];
        sigma[i]   = sigma_global[i];
    }
    __syncthreads();

    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_local ; i += gridDim.x * blockDim.x ) {
        int n = nbond[i];
        f3i coord1 = tex1Dfetch<float4>( tex_coord_merged, i );

        r64 fx = 0.0, fy = 0.0, fz = 0.0;
        r64 e = 0.0 ;

        for( int p = 0 ; p < n ; p++ ) {
            int j    = bonds[ i + p * padding ].x;
            int type = bonds[ i + p * padding ].y;

            f3i coord2 = tex1Dfetch<float4>( tex_coord_merged, j );
            r64 dx = minimum_image( coord1.x - coord2.x, period.x ) ;
            r64 dy = minimum_image( coord1.y - coord2.y, period.y ) ;
            r64 dz = minimum_image( coord1.z - coord2.z, period.z ) ;

            r64 rsq   = dx * dx + dy * dy + dz * dz ;

            r64 r0sq    = r0[type] * r0[type];
            r64 rlogarg = 1.0 - rsq / r0sq;

            // if r -> r0, then rlogarg < 0.0 which is an error
            // issue a warning and reset rlogarg = epsilon
            // if r > 2*r0 something serious is wrong, abort

            rlogarg = max( 0.1, rlogarg );
            r64 fbond = -k[type] / rlogarg;

            if( evflag ) e += -0.5 * k[type] * r0sq * log( rlogarg );

            // force & energy from LJ term

            if( rsq < TWO_1_3 * sigma[type] * sigma[type] ) {
                r64 sr2 = sigma[type] * sigma[type] / rsq;
                r64 sr6 = sr2 * sr2 * sr2;
                fbond += 48.0 * epsilon[type] * sr6 * ( sr6 - 0.5 ) / rsq;
                if( evflag ) e += 4.0 * epsilon[type] * sr6 * ( sr6 - 1.0 ) + epsilon[type];
            }

            fx += dx * fbond ;
            fy += dy * fbond ;
            fz += dz * fbond ;
        }

        force_x[i] += fx;
        force_y[i] += fy;
        force_z[i] += fz;
        if( evflag ) {
            virial_xx[i] += coord1.x * fx ;
            virial_yy[i] += coord1.y * fy ;
            virial_zz[i] += coord1.z * fz ;
            virial_xy[i] += coord1.x * fy ;
            virial_xz[i] += coord1.x * fz ;
            virial_yz[i] += coord1.y * fz ;
            e_bond[i]     = e * 0.5;
        }
    }
}

void MesoBondFENE::compute( int eflag, int vflag )
{
    if( !coeff_alloced ) alloc_coeff();

    static GridConfig grid_cfg, grid_cfg_EV;
    if( !grid_cfg_EV.x ) {
        grid_cfg_EV = meso_device->occu_calc.right_peak( 0, gpu_bond_fene<1>, ( atom->nbondtypes + 1 ) * 2 * sizeof( r64 ), cudaFuncCachePreferL1 );
        cudaFuncSetCacheConfig( gpu_bond_fene<1>, cudaFuncCachePreferL1 );
        grid_cfg    = meso_device->occu_calc.right_peak( 0, gpu_bond_fene<0>, ( atom->nbondtypes + 1 ) * 2 * sizeof( r64 ), cudaFuncCachePreferL1 );
        cudaFuncSetCacheConfig( gpu_bond_fene<0>, cudaFuncCachePreferL1 );
    }

    double3 period;
    period.x = ( domain->xperiodic ) ? ( domain->xprd ) : ( 0. );
    period.y = ( domain->yperiodic ) ? ( domain->yprd ) : ( 0. );
    period.z = ( domain->zperiodic ) ? ( domain->zprd ) : ( 0. );

    if( eflag || vflag ) {
        gpu_bond_fene<1> <<< grid_cfg_EV.x, grid_cfg_EV.y, ( atom->nbondtypes + 1 ) * 2 * sizeof( r64 ), meso_device->stream() >>> (
            meso_atom->tex_coord_merged,
            meso_atom->dev_force(0),
            meso_atom->dev_force(1),
            meso_atom->dev_force(2),
            meso_atom->dev_virial(0),
            meso_atom->dev_virial(1),
            meso_atom->dev_virial(2),
            meso_atom->dev_virial(3),
            meso_atom->dev_virial(4),
            meso_atom->dev_virial(5),
            meso_atom->dev_e_bond,
            meso_atom->dev_nbond,
            meso_atom->dev_bond_mapped,
            dev_k,
            dev_r0,
            dev_epsilon,
            dev_sigma,
            period,
            meso_atom->dev_bond_mapped.pitch_elem(),
            atom->nbondtypes,
            atom->nlocal );
    } else {
        gpu_bond_fene<0> <<< grid_cfg.x, grid_cfg.y, ( atom->nbondtypes + 1 ) * 2 * sizeof( r64 ), meso_device->stream() >>> (
            meso_atom->tex_coord_merged,
            meso_atom->dev_force(0),
            meso_atom->dev_force(1),
            meso_atom->dev_force(2),
            meso_atom->dev_virial(0),
            meso_atom->dev_virial(1),
            meso_atom->dev_virial(2),
            meso_atom->dev_virial(3),
            meso_atom->dev_virial(4),
            meso_atom->dev_virial(5),
            meso_atom->dev_e_bond,
            meso_atom->dev_nbond,
            meso_atom->dev_bond_mapped,
            dev_k,
            dev_r0,
            dev_epsilon,
            dev_sigma,
            period,
            meso_atom->dev_bond_mapped.pitch_elem(),
            atom->nbondtypes,
            atom->nlocal );
    }
}

