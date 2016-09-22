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
#include "fix_boundary_fc_trp_special_meso.h"
#include "pair_edpd_trp_base_meso.h"

using namespace LAMMPS_NS;
using namespace PNIPAM_COEFFICIENTS;

/* ---------------------------------------------------------------------- */

MesoFixBoundaryFcTRPSpecial::MesoFixBoundaryFcTRPSpecial( LAMMPS *lmp, int narg, char **arg ):
    Fix( lmp, narg, arg ),
    MesoPointers( lmp ),
    wall_type(1),
    cut( 0. ),
    cx( 0. ), cy( 0. ), cz( 0. ),
    ox( 0. ), oy( 0. ), oz( 0. ),
    radius( 0. ), length( 0. ),
    T_H( 1. ), T_C( 1. ),
    poly( lmp, "MesoFixBoundaryFcTRPSpecial::poly" ),
	pair(NULL)
{
    pair = dynamic_cast<MesoPairEDPDTRPBase*>( force->pair );
    if( !pair ) error->all( FLERR, "<MESO> fix boundary/fc/trp/meso must be used together with pair edpd/pnipam/meso" );

    for( int i = 3; i < narg; i++ ) {
        if( !strcmp( arg[i], "type" ) ) {
            wall_type = atoi( arg[++i] );
            continue;
        }
        if( !strcmp( arg[i], "cut" ) ) {
            cut = atof( arg[++i] );
            continue;
        }
        if( !strcmp( arg[i], "T" ) ) {
            T_H = atof( arg[++i] );
            T_C = atof( arg[++i] );
            continue;
        }
        if( !strcmp( arg[i], "radius" ) ) {
            radius = atof( arg[++i] );
            continue;
        }
        if( !strcmp( arg[i], "length" ) ) {
            length = atof( arg[++i] );
            continue;
        }
        if( !strcmp( arg[i], "center" ) ) {
            cx = atof( arg[++i] );
            cy = atof( arg[++i] );
            cz = atof( arg[++i] );
            continue;
        }
        if( !strcmp( arg[i], "orient" ) ) {
            ox = atof( arg[++i] );
            oy = atof( arg[++i] );
            oz = atof( arg[++i] );
            continue;
        }
        if( !strcmp( arg[i], "poly" ) ) {
            int order = atoi( arg[++i] );
            poly.grow( order + 1 );
            for( int j = 0; j < order + 1; j++ ) poly[j] = atof( arg[++i] );
            continue;
        }
    }

    if( ( ox == 0. && oy == 0. && oz == 0. ) || radius < 1 || poly.n() == 0 || poly == NULL )
        error->all( FLERR, "Usage: boundary/fc group [type int] [T0 double] [cut double] [radius double] [length double] [center doublex3] [orient doublex3] [poly int doublex?]" );

    double n = std::sqrt( ox * ox + oy * oy + oz * oz );
    ox /= n;
    oy /= n;
    oz /= n;
}

MesoFixBoundaryFcTRPSpecial::~MesoFixBoundaryFcTRPSpecial()
{
}

int MesoFixBoundaryFcTRPSpecial::setmask()
{
    int mask = 0;
    mask |= FixConst::POST_FORCE;
    return mask;
}

void MesoFixBoundaryFcTRPSpecial::init()
{
    if( strcmp( update->integrate_style, "respa" ) == 0 ) {
        fprintf( stderr, "<MESO> RESPA not supported in MesoFixBoundaryFcTRPSpecial. %s %cut\n", __FILE__, __LINE__ );
    }
}

void MesoFixBoundaryFcTRPSpecial::setup( int vflag )
{
    if( strcmp( update->integrate_style, "respa" ) == 0 ) {
        fprintf( stderr, "<MESO> RESPA not supported in MesoFixBoundaryFcTRPSpecial. %s %cut\n", __FILE__, __LINE__ );
    }
    post_force( vflag );
}

__global__ void gpu_fix_boundary_fc_trp_special(
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
    const r64 T_H,
    const r64 T_C,
    const r64 cx,
    const r64 cy,
    const r64 cz,
    const r64 ox,
    const r64 oy,
    const r64 oz,
    const r64 cut,
    const r64 radius,
    const r64 length,
    const int n )
{
    extern __shared__ r64 coeffs[];
    for( int p = threadIdx.x; p < n_type * n_type * n_coeff; p += blockDim.x )
        coeffs[p] = coefficients[p];
    __syncthreads();

    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x ) {
        if( mask[i] & groupbit ) {
        	r64 dx = coord_x[i] - cx;
        	r64 dy = coord_y[i] - cy;
        	r64 dz = coord_z[i] - cz;
        	r64 along = dx * ox + dy * oy + dz * oz;
        	r64 perpx = dx - along * ox;
        	r64 perpy = dy - along * oy;
        	r64 perpz = dz - along * oz;
        	r64 d = sqrt( perpx*perpx + perpy*perpy + perpz*perpz );
            if( d > radius - cut ) {
                r64 T0        = ( fmod( along, length ) > ( length * 0.5 ) ) ? T_C : T_H;
                r64 h         = max( min( radius - d, cut ), 0. );
                r64 wc        = max( polyval( h, order, poly ), 0. );
                r64 *coeff_ij = coeffs + ( (type[i]-1) * n_type + (wall_type-1) ) * n_coeff;
                r64 T_ij      = 0.5 * ( T[i] + T0 );
                r64 a_0       = coeff_ij[p_a0] * T_ij;
                r64 a_inc     = coeff_ij[p_da] / ( 1.0 + expf( coeff_ij[p_omega] * ( T_ij - coeff_ij[p_theta]  ) ) );
                r64 force     = wc * ( a_0 + a_inc );
                force_x[i]   -= force * perpx / d;
                force_y[i]   -= force * perpy / d;
                force_z[i]   -= force * perpz / d;
            }
        }
    }
}

void MesoFixBoundaryFcTRPSpecial::post_force( int vflag )
{
    static GridConfig grid_cfg;
    if( !grid_cfg.x ) {
        grid_cfg = meso_device->occu_calc.right_peak( 0, gpu_fix_boundary_fc_trp_special, 0, cudaFuncCachePreferL1 );
        cudaFuncSetCacheConfig( gpu_fix_boundary_fc_trp_special, cudaFuncCachePreferL1 );
    }

    prepare_coeff();

    gpu_fix_boundary_fc_trp_special <<< grid_cfg.x, grid_cfg.y, pair->dev_coefficients.size(), meso_device->stream() >>> (
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
        T_H, T_C,
        cx, cy, cz,
        ox, oy, oz,
        cut, radius, length,
        atom->nlocal );
}

void MesoFixBoundaryFcTRPSpecial::prepare_coeff() {
	pair->prepare_coeff();
}
