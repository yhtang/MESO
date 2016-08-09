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
#include "fix_boundary_qc_special_meso.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

MesoFixBoundaryQcSpecial::MesoFixBoundaryQcSpecial( LAMMPS *lmp, int narg, char **arg ):
    Fix( lmp, narg, arg ),
    MesoPointers( lmp ),
    cut( 0. ),
    cx( 0. ), cy( 0. ), cz( 0. ),
    ox( 0. ), oy( 0. ), oz( 0. ),
    radius( 0. ), length( 0. ),
    T_H( 1. ), T_C( 1. ), a0( 0. ),
    poly( lmp, "MesoFixBoundaryQcSpecial::poly" )
{
    for( int i = 3; i < narg; i++ ) {
        if( !strcmp( arg[i], "cut" ) ) {
            cut = atof( arg[++i] );
            continue;
        }
        if( !strcmp( arg[i], "T" ) ) {
            T_H = atof( arg[++i] );
            T_C = atof( arg[++i] );
            continue;
        }
        if( !strcmp( arg[i], "a0" ) ) {
            a0 = atof( arg[++i] );
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

    if( ( ox == 0. && oy == 0. && oz == 0. ) || radius < 1 || poly.n() == 0 || a0 == 0. )
        error->all( FLERR, "Usage: boundary/fc group [cut double] [T doublex2] [a0 double] [radius double] [length double] [center doublex3] [orient doublex3] [poly int doublex?]" );

    double n = std::sqrt( ox * ox + oy * oy + oz * oz );
    ox /= n;
    oy /= n;
    oz /= n;
}

MesoFixBoundaryQcSpecial::~MesoFixBoundaryQcSpecial()
{
}

int MesoFixBoundaryQcSpecial::setmask()
{
    int mask = 0;
    mask |= FixConst::POST_FORCE;
    return mask;
}

void MesoFixBoundaryQcSpecial::init()
{
    if( strcmp( update->integrate_style, "respa" ) == 0 ) {
        fprintf( stderr, "<MESO> RESPA not supported in MesoFixBoundaryQcSpecial. %s %cut\n", __FILE__, __LINE__ );
    }
}

void MesoFixBoundaryQcSpecial::setup( int vflag )
{
    if( strcmp( update->integrate_style, "respa" ) == 0 ) {
        fprintf( stderr, "<MESO> RESPA not supported in MesoFixBoundaryQcSpecial. %s %cut\n", __FILE__, __LINE__ );
    }
    post_force( vflag );
}

__global__ void gpu_fix_boundary_qc_special(
    r64* __restrict coord_x,
    r64* __restrict coord_y,
    r64* __restrict coord_z,
    r64* __restrict T,
    r64* __restrict Q,
    int* __restrict mask,
    const int groupbit,
    const int order,
    r64* __restrict poly,
    const r64 cx,
    const r64 cy,
    const r64 cz,
    const r64 ox,
    const r64 oy,
    const r64 oz,
    const r64 cut,
    const r64 radius,
    const r64 length,
    const r64 T_H,
    const r64 T_C,
    const r64 a0,
    const int n,
    const int timestamp )
{
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
                r64 T0  = ( fmod( along, length ) > ( length * 0.5 ) ) ? T_C : T_H;
                r64 h   = max( min( radius - d, cut ), 0. );
                r64 qc  = a0 * max( polyval( h, order, poly ), 0. );
                r64 sig = sqrt( 2. * qc ) * ( T[i] + T0 );
                r64 rn  = gaussian_TEA<16>( true, timestamp, i );
                r64 qr  = sig * rn;
                qc     *= power<2>( T[i] + T0 ) * ( 1./T[i] - 1./T0 );
                Q[i]   += qc + qr;
            }
        }
    }
}

void MesoFixBoundaryQcSpecial::post_force( int vflag )
{
    static GridConfig grid_cfg;
    if( !grid_cfg.x ) {
        grid_cfg = meso_device->occu_calc.right_peak( 0, gpu_fix_boundary_qc_special, 0, cudaFuncCachePreferL1 );
        cudaFuncSetCacheConfig( gpu_fix_boundary_qc_special, cudaFuncCachePreferL1 );
    }

    gpu_fix_boundary_qc_special <<< grid_cfg.x, grid_cfg.y, 0, meso_device->stream() >>> (
        meso_atom->dev_coord[0], meso_atom->dev_coord[1], meso_atom->dev_coord[2],
        meso_atom->dev_T, meso_atom->dev_Q,
        meso_atom->dev_mask,
        groupbit,
        poly.n() - 1,
        poly,
        cx, cy, cz,
        ox, oy, oz,
        cut, radius, length,
        T_H, T_C, a0,
        atom->nlocal,
        update->ntimestep );
}
