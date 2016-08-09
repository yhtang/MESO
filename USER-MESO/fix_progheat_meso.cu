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
#include "fix_progheat_meso.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

MesoFixProgHeat::MesoFixProgHeat( LAMMPS *lmp, int narg, char **arg ):
    Fix( lmp, narg, arg ),
    MesoPointers( lmp ),
    t_user("@t"),
    T_varname("__PNIPAM_internal_programmable_temperature__"),
	hst_meanT(lmp,"MesoFixProgHeat::hst_meanT"),
	dev_dQ(lmp,"MesoFixProgHeat::dev_dQ")
{
    if( narg < 5 ) error->all( FLERR, "Illegal fix CUDAPoiseuille command" );

    scalar_flag = 1;
    extscalar = 0;
    global_freq = 1;

    T_command = new char*[3];
    for(int i = 0 ; i < 3 ; i++) T_command[i] = new char[128];
    strcpy( T_command[0], T_varname.c_str() );
    strcpy( T_command[1], "equal" );

    delta = 0.;
    Cv = 0.;

    for( int i = 3; i < narg; i++ ) {
        if( !strcmp( arg[i], "T" ) ) {
        	T_template = arg[++i];
            continue;
        }
        if( !strcmp( arg[i], "delta" ) ) {
            delta = atof( arg[++i] );
            continue;
        }
        if( !strcmp( arg[i], "Cv" ) ) {
            Cv = atof( arg[++i] );
            continue;
        }
    }

    hst_meanT.grow(1);
    dev_dQ.grow(1);

    if ( T_template == "" ) error->all( FLERR, "<MESO> progheat/meso usage: [Cv double] [T expression] [delta double]" );
    if ( Cv == 0. ) error->warning( FLERR, "<MESO> progheat/meso Cv = 0, cannot calculate heat flux" );
}

MesoFixProgHeat::~MesoFixProgHeat() {
    for(int i=0;i<3;i++) delete [] T_command[i];
    delete [] T_command;
}

int MesoFixProgHeat::setmask()
{
    return FixConst::POST_FORCE;
}

void MesoFixProgHeat::init()
{
    if( strcmp( update->integrate_style, "respa" ) == 0 ) {
        fprintf( stderr, "<MESO> RESPA not supported in MesoFixProgHeat. %s %d\n", __FILE__, __LINE__ );
    }
}

void MesoFixProgHeat::setup( int vflag )
{
    if( strcmp( update->integrate_style, "respa" ) == 0 ) {
        fprintf( stderr, "<MESO> RESPA not supported in MesoFixProgHeat. %s %d\n", __FILE__, __LINE__ );
    }
    post_force( vflag );
}

__global__ void gpu_fix_prog_heat(
    r64* __restrict T,
    int* __restrict mask,
    r32* __restrict dQ,
    const r64 delta,
    const r64 Cv,
    const r64 T0,
    const int groupbit,
    const int n )
{
    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x ) {
        if( mask[i] & groupbit ) {
        	r64 dT = ( T0 - T[i] ) * delta;
        	T[i] += dT;
        	atomic_add( dQ, dT * Cv );
        }
    }
}

void MesoFixProgHeat::post_force( int vflag )
{
    static GridConfig grid_cfg;
    if( !grid_cfg.x ) {
        grid_cfg = meso_device->occu_calc.right_peak( 0, gpu_fix_prog_heat, 0, cudaFuncCachePreferL1 );
        cudaFuncSetCacheConfig( gpu_fix_prog_heat, cudaFuncCachePreferL1 );
    }

    // calculate current time
    double t = double( update->ntimestep ) * update->dt;
    // substitute time into temperature function
    char t_current[256];
    sprintf( t_current, "%le", t );
    std::string t_str( t_current );
    std::string T_expr( T_template );
    while( T_expr.find(t_user) != std::string::npos ) {
    	T_expr.replace( T_expr.find(t_user), t_user.size(), t_str );
    }
    // update LAMMPS variable for broadcasting
    strcpy( T_command[2], T_expr.c_str() );
    input->variable->set( 3, T_command );
    // evaluate
    double T0 = input->variable->compute_equal( input->variable->find( (char*)T_varname.c_str() ) );
    /*if ( update->ntimestep%100==0) {
    	fprintf( stderr, "%s = %12.15lf\n", T_expr.c_str(), T0 );
		gpu_reduce_sum_host<double><<<1,1024,0,meso_device->stream()>>>( meso_atom->dev_T, meanT.ptr(), atom->nlocal );
		meso_device->sync_device();
		printf("mean T = %.4lf\n", meanT.ptr()[0] / atom->nlocal );
    }*/

    dev_dQ.set( 0.f, meso_device->stream() );
    gpu_fix_prog_heat <<< grid_cfg.x, grid_cfg.y, 0, meso_device->stream() >>> (
        meso_atom->dev_T,
        meso_atom->dev_mask,
        dev_dQ,
        delta,
        Cv,
        T0,
        groupbit,
        atom->nlocal );
}

double MesoFixProgHeat::compute_scalar()
{
    r32 dQ = 0.;
    dev_dQ.download( &dQ, 1, meso_device->stream() );
    meso_device->sync_device();

    double dQ_double = dQ;
    double scalar;
    MPI_Allreduce( &dQ_double, &scalar, 1, MPI_DOUBLE, MPI_SUM, world );
    return scalar;
}
