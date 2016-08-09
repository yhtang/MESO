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

#include "mpi.h"
#include "stdio.h"
#include "string.h"
#include "force.h"
#include "update.h"
#include "error.h"
#include "domain.h"
#include "input.h"
#include "variable.h"

#include "atom_meso.h"
#include "comm_meso.h"
#include "atom_vec_meso.h"
#include "engine_meso.h"
#include "fix_boundary_fc_trp_meso.h"
#include "pair_edpd_trp_base_meso.h"

using namespace LAMMPS_NS;
using namespace PNIPAM_COEFFICIENTS;

/* ---------------------------------------------------------------------- */

MesoFixBoundaryFcTRP::MesoFixBoundaryFcTRP( LAMMPS *lmp, int narg, char **arg ):
    Fix( lmp, narg, arg ),
    MesoPointers( lmp ),
    wall_type(1),
    cut( 0. ),
    H( 0. ),
    nx( 0. ), ny( 0. ), nz( 0. ),
    poly( lmp, "MesoFixBoundaryFcTRP::poly" ),
	pair(NULL)
{
    pair = dynamic_cast<MesoPairEDPDTRPBase*>( force->pair );
    if( !pair ) error->all( FLERR, "<MESO> fix boundary/fc/trp/meso must be used together with pair edpd/pnipam/meso" );

    bool set_H_by_example_point = false;
    double px, py, pz;

    for( int i = 3; i < narg; i++ ) {
        if( !strcmp( arg[i], "type" ) ) {
            wall_type = atoi( arg[++i] );
            continue;
        }
        if( !strcmp( arg[i], "cut" ) ) {
            cut = atof( arg[++i] );
            continue;
        }
        if( !strcmp( arg[i], "T0" ) ) {
            T0 = atof( arg[++i] );
            continue;
        }
        if( !strcmp( arg[i], "H" ) ) {
            H = atof( arg[++i] );
            continue;
        }
        if( !strcmp( arg[i], "p" ) ) {
            px = atof( arg[++i] );
            py = atof( arg[++i] );
            pz = atof( arg[++i] );
            set_H_by_example_point = true;
            continue;
        }
        if( !strcmp( arg[i], "n" ) ) {
            nx = atof( arg[++i] );
            ny = atof( arg[++i] );
            nz = atof( arg[++i] );
            continue;
        }
        if( !strcmp( arg[i], "poly" ) ) {
            int order = atoi( arg[++i] );
            poly.grow( order + 1 );
            for( int j = 0; j < order + 1; j++ ) poly[j] = atof( arg[++i] );
            continue;
        }
    }

    if( ( nx == 0. && ny == 0. && nz == 0. ) || poly.n() == 0 || poly == NULL )
        error->all( FLERR, "Usage: boundary/fc group [type int] [T0 double] [cut double] [H double]|[p doublex3] [n doublex3] [poly int doublex?]" );

    double n = std::sqrt( nx * nx + ny * ny + nz * nz );
    nx /= n;
    ny /= n;
    nz /= n;

    if( set_H_by_example_point ) {
        H = px * nx + py * ny + pz * nz;
    }
}

MesoFixBoundaryFcTRP::~MesoFixBoundaryFcTRP()
{
}

int MesoFixBoundaryFcTRP::setmask()
{
    int mask = 0;
    mask |= FixConst::POST_FORCE;
    return mask;
}

void MesoFixBoundaryFcTRP::init()
{
    if( strcmp( update->integrate_style, "respa" ) == 0 ) {
        fprintf( stderr, "<MESO> RESPA not supported in MesoFixBoundaryFcTRP. %s %cut\n", __FILE__, __LINE__ );
    }
}

void MesoFixBoundaryFcTRP::setup( int vflag )
{
    if( strcmp( update->integrate_style, "respa" ) == 0 ) {
        fprintf( stderr, "<MESO> RESPA not supported in MesoFixBoundaryFcTRP. %s %cut\n", __FILE__, __LINE__ );
    }
    post_force( vflag );
}

__global__ void gpu_fix_boundary_fc_trp(
    r64* __restrict coord_x,
    r64* __restrict coord_y,
    r64* __restrict coord_z,
    r64* __restrict force_x,
    r64* __restrict force_y,
    r64* __restrict force_z,
    r64* __restrict T,
    int* __restrict type,
    int* __restrict mask,
    r64* __restrict coefficients,
    const int n_type,
    const int wall_type,
    const int groupbit,
    const int order,
    r64* __restrict poly,
    const r64 T0,
    const r64 nx,
    const r64 ny,
    const r64 nz,
    const r64 cut,
    const r64 H,
    const int n )
{
    extern __shared__ r64 coeffs[];
    for( int p = threadIdx.x; p < n_type * n_type * n_coeff; p += blockDim.x )
        coeffs[p] = coefficients[p];
    __syncthreads();

    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x ) {
        if( mask[i] & groupbit ) {
            r64 d = coord_x[i] * nx + coord_y[i] * ny + coord_z[i] * nz;
            if( d > H - cut ) {
                r64 h         = max( min( H - d, cut ), 0. );
                r64 wc        = max( polyval( h, order, poly ), 0. );
                r64 *coeff_ij = coeffs + ( (type[i]-1) * n_type + (wall_type-1) ) * n_coeff;
                r64 T_ij      = 0.5 * ( T[i] + T0 );
                r64 a_0       = coeff_ij[p_a0] * T_ij;
                r64 a_inc     = coeff_ij[p_da] / ( 1.0 + expf( coeff_ij[p_omega] * ( T_ij - coeff_ij[p_theta]  ) ) );
                r64 force     = wc * ( a_0 + a_inc );
                force_x[i]   += force * -nx;
                force_y[i]   += force * -ny;
                force_z[i]   += force * -nz;
            }
        }
    }
}

void MesoFixBoundaryFcTRP::post_force( int vflag )
{
    static GridConfig grid_cfg;
    if( !grid_cfg.x ) {
        grid_cfg = meso_device->occu_calc.right_peak( 0, gpu_fix_boundary_fc_trp, 0, cudaFuncCachePreferL1 );
        cudaFuncSetCacheConfig( gpu_fix_boundary_fc_trp, cudaFuncCachePreferL1 );
    }

    prepare_coeff();

    double T = T0;
    int var = input->variable->find( (char*)("__PNIPAM_internal_programmable_temperature__") );
    if ( var != -1 ) {
    	T = input->variable->compute_equal(var);
    }

    gpu_fix_boundary_fc_trp <<< grid_cfg.x, grid_cfg.y, pair->dev_coefficients.size(), meso_device->stream() >>> (
        meso_atom->dev_coord[0], meso_atom->dev_coord[1], meso_atom->dev_coord[2],
        meso_atom->dev_force[0], meso_atom->dev_force[1], meso_atom->dev_force[2],
        meso_atom->dev_T,
        meso_atom->dev_type, meso_atom->dev_mask,
        pair->dev_coefficients,
        atom->ntypes,
        wall_type,
        groupbit,
        poly.n() - 1,
        poly,
        T,
        nx, ny, nz,
        cut, H,
        atom->nlocal );
}

void MesoFixBoundaryFcTRP::prepare_coeff() {
	pair->prepare_coeff();
}
