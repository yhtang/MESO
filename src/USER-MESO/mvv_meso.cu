#include "domain.h"
#include "force.h"
#include "pair.h"
#include "bond.h"
#include "angle.h"
#include "dihedral.h"
#include "improper.h"
#include "kspace.h"
#include "output.h"
#include "update.h"
#include "modify.h"
#include "compute.h"
#include "fix.h"
#include "timer.h"
#include "memory.h"
#include "error.h"

#include "atom_meso.h"
#include "atom_vec_meso.h"
#include "comm_meso.h"
#include "engine_meso.h"
#include "mvv_meso.h"
#include "neighbor_meso.h"
#include "math_meso.h"
#include "timer_meso.h"
#include "nvtx_meso.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ModifiedVerlet::ModifiedVerlet( LAMMPS *lmp, int narg, char **arg ) :
    Integrate( lmp, narg, arg ), MesoPointers( lmp )
{
}

/* ----------------------------------------------------------------------
helper function for checking CUDA runtime error
---------------------------------------------------------------------- */

void ModifiedVerlet::check_error( int linenum, const char filename[] )
{
//#define LMP_MESO_DEBUG
#ifdef LMP_MESO_DEBUG
	cudaError_t err = cudaDeviceSynchronize();
    MPI_Barrier( MPI_COMM_WORLD );
    char error_info[512];
    sprintf( error_info, "[CDBG] %s [ line %d @rank %d ]\n", cudaGetErrorString( err ), linenum, comm->me );
    if( comm->me == 0 ) {
        MPI_Status stat;
        fprintf( stderr, error_info );
        for( int i = 1 ; i < comm->nprocs ; i++ ) {
            MPI_Recv( error_info, 512, MPI_CHAR, i, i, MPI_COMM_WORLD, &stat );
            fprintf( stderr, error_info );
        }
    } else {
        MPI_Send( error_info, 512, MPI_CHAR, 0, comm->me, MPI_COMM_WORLD );
    }
    MPI_Barrier( MPI_COMM_WORLD );
    if( cudaPeekAtLastError() ) {
        cudaDeviceReset();
        error->one( filename, linenum, error_info );
        exit( 0 );
    }
#else
//  static bool foo = true;
//  cuda_engine->sync_device();
//  if (foo) {
//      error->warning( FLERR, "syncing debug ON" );
//      foo = false;
//  }
#endif
}

/* ----------------------------------------------------------------------
     initialization before run
------------------------------------------------------------------------- */

void ModifiedVerlet::init()
{
    Integrate::init();

    // warn if no fixes

    if( modify->nfix == 0 && comm->me == 0 )
        error->warning( FLERR, "No fixes defined, atoms won't move" );

    // virial_style:
    // 1 if computed explicitly by pair->compute via sum over pair interactions
    // 2 if computed implicitly by pair->virial_compute via sum over ghost atoms

    if( force->newton_pair ) virial_style = 2;
    else virial_style = 1;

    // setup lists of computes for global and per-atom PE and pressure

    ev_setup();

    // check for compliance with USER-MESO
    char warning_info[512];
    if( force->newton_pair || force->newton || force->newton_bond ) {
        if( comm->me == 0 ) {
            sprintf( warning_info, "<MESO> newton_pair not allowed in MESO mode, forced to 0; ghost_velocity forced to 1" );
            error->warning( FLERR, warning_info );
        }
        force->newton      = 0;
        force->newton_pair = 0;
        force->newton_bond = 0;
        comm->ghost_velocity = 1;
    }
    if( virial_style == 2 ) {
        if( comm->me == 0 ) {
            sprintf( warning_info, "<MESO> implicit virial computation not allowed in MESO mode, forced to explicit" );
            error->warning( FLERR, warning_info );
        }
        virial_style = 1 ;
    }
    if( domain->triclinic ) {
        sprintf( warning_info, "<MESO> triclinic domain not supported in USER-MESO" );
        error->one( FLERR, warning_info );
    }
    if( atom->sortfreq > 0 ) {
        if( comm->me == 0 ) {
            sprintf( warning_info, "<MESO> atom sort frequency managed automatically in USER-MESO" );
            error->warning( FLERR, warning_info );
        }
        atom->sortfreq = 0;
    }
    if( force->kspace ) {
        sprintf( warning_info, "<MESO> kspace not supported in USER-MESO" );
        error->one( FLERR, warning_info );
    }
}

/* ----------------------------------------------------------------------
     setup before run
------------------------------------------------------------------------- */

void ModifiedVerlet::setup()
{
    if( comm->me == 0 && screen ) fprintf( screen, "Setting up run ...\n" );

    update->setupflag = 1;

    // start the profiler if specified profiling mode
    meso_device->configure_profiler( update->ntimestep, update->nsteps );

    // setup domain, communication and neighboring
    // acquire ghosts
    // build neighbor lists

    atom->setup();
    modify->setup_pre_exchange();
    domain->pbc();
    domain->reset_box();
    comm->setup();
    if( neighbor->style ) neighbor->setup_bins();
    comm->exchange();

    check_error( __LINE__, __FILE__ );

    meso_atom->transfer_pre_sort();
    meso_atom->transfer_pre_post_sort();
    meso_atom->sort_local(); // atom data reorder
    meso_atom->transfer_post_sort();
    check_error( __LINE__, __FILE__ );

    meso_atom->transfer_pre_border();
    comm->borders();
    meso_atom->transfer_post_border();
    meso_atom->map_set_device();

    check_error( __LINE__, __FILE__ );

    meso_atom->transfer_pre_output(); // otherwise the *check functions might be checking stale data
    check_error( __LINE__, __FILE__ );
    atom->map_set();
    check_error( __LINE__, __FILE__ );
    domain->image_check();
    check_error( __LINE__, __FILE__ );
    domain->box_too_small_check();
    check_error( __LINE__, __FILE__ );
    modify->setup_pre_neighbor();
    check_error( __LINE__, __FILE__ );
    neighbor->build();
    check_error( __LINE__, __FILE__ );
    neighbor->ncalls = 0;
    check_error( __LINE__, __FILE__ );

    check_error( __LINE__, __FILE__ );

    // compute all forces
    ev_set( update->ntimestep );
    force_clear();

    check_error( __LINE__, __FILE__ );

    modify->setup_pre_force( vflag );
    if( pair_compute_flag ) force->pair->compute( eflag, vflag );
    else if( force->pair ) force->pair->compute_dummy( eflag, vflag );

    check_error( __LINE__, __FILE__ );

    if( atom->molecular ) {
        if( force->bond ) force->bond->compute( eflag, vflag );
        if( force->angle ) force->angle->compute( eflag, vflag );
        if( force->dihedral ) force->dihedral->compute( eflag, vflag );
        if( force->improper ) force->improper->compute( eflag, vflag );
    }

    check_error( __LINE__, __FILE__ );

    modify->setup( vflag );
    meso_atom->transfer_pre_output();
    output->setup();
    update->setupflag = 0;

    check_error( __LINE__, __FILE__ );
}

/* ----------------------------------------------------------------------
   setup without output
   flag = 0 = just force calculation
   flag = 1 = reneighbor and force calculation
------------------------------------------------------------------------- */

void ModifiedVerlet::setup_minimal( int flag )
{
    setup();
}

/* ----------------------------------------------------------------------
     iterate for n steps
------------------------------------------------------------------------- */
template<template<typename, typename> class CONTAINER, typename T>
T mean( const CONTAINER<T, std::allocator<T> > &c )
{
    T sum = T( 0 );
    for( size_t i = 0; i < c.size(); i++ ) sum += c[i];
    return sum / c.size();
}

void ModifiedVerlet::run( int n )
{
    bigint ntimestep;

    int n_post_integrate = modify->n_post_integrate;
    int n_pre_exchange = modify->n_pre_exchange;
    int n_pre_neighbor = modify->n_pre_neighbor;
    int n_pre_force = modify->n_pre_force;
    int n_post_force = modify->n_post_force;
    int n_end_of_step = modify->n_end_of_step;

    meso_device->set_device();
    for( int i = 0; i < n; i++ ) {
    	ntimestep = ++update->ntimestep;
        ev_set( ntimestep );

        meso_device->configure_profiler( update->ntimestep, update->nsteps );

        // initial time integration
        nvtx_push_range( "InitInt", nvtx_color(0) );
        modify->initial_integrate( vflag );
        if( n_post_integrate ) modify->post_integrate();
        nvtx_pop_range( "InitInt" );

        check_error( __LINE__, __FILE__ );

        // regular communication vs neighbor list rebuild
        if ( neighbor->decide() )
        {
        	nvtx_push_range( "PreXchg", nvtx_color(1) );
        	if( n_pre_exchange ) modify->pre_exchange();
            nvtx_pop_range( "PreXchg" );

        	nvtx_push_range( "\u2198Xchg", nvtx_color(2) );
            meso_atom->transfer_pre_exchange();
        	nvtx_pop_range( "\u2198Xchg" );

        	//******************************************//
        	nvtx_push_range( "\u26D6Xchg", nvtx_color(2) );

            domain->pbc();                              // wrap_molecules
            if( domain->box_change ) {
                domain->reset_box();
                comm->setup();
                if( neighbor->style ) neighbor->setup_bins();
            }
            timer->stamp();
            comm->exchange();

        	nvtx_pop_range( "\u26D6Xchg" );
            //******************************************//

        	nvtx_push_range( "\u2197Sort", nvtx_color(3) );
            meso_atom->transfer_pre_sort();
            meso_atom->transfer_pre_post_sort();
            nvtx_pop_range( "\u2197Sort" );
        	nvtx_push_range( "\u26F1Sort", nvtx_color(3) );
            meso_atom->sort_local(); // atom data reorder
            nvtx_pop_range( "\u26F1Sort" );
        	nvtx_push_range( "\u27B6Sort", nvtx_color(3) );
            meso_atom->transfer_post_sort();
            nvtx_pop_range( "\u27B6sort" );


            nvtx_push_range( "\u2198Border", nvtx_color(4) );

            meso_atom->transfer_pre_border();
            nvtx_pop_range( "\u2198Border" );
            nvtx_push_range( "\u26D6Border", nvtx_color(4) );
            comm->borders();
            nvtx_pop_range( "\u26D6Border" );
            nvtx_push_range( "\u2197Border", nvtx_color(4) );
            meso_atom->transfer_post_border();
            meso_atom->map_set_device();
            nvtx_pop_range( "\u2197Border" );

            nvtx_push_range( "PreNeigh", nvtx_color(5) );
            timer->stamp( TIME_COMM );
            if( n_pre_neighbor ) modify->pre_neighbor();
            nvtx_pop_range( "PreNeigh" );

            nvtx_push_range( "\u9130Neighbor", nvtx_color(6) );
            neighbor->build();
            nvtx_pop_range( "\u9130Neighbor" );

            timer->stamp( TIME_NEIGHBOR );

            if( force->pair->split_flag && pair_compute_flag ) {
        	    nvtx_push_range( "\u26EFForceB", nvtx_color(7) );
        	    force_clear( AtomAttribute::LOCAL );
        	    force->pair->compute_bulk( eflag, vflag );
                nvtx_pop_range( "\u26EFForceB" );
            }
        } else {
        	nvtx_push_range( "\u2198Comm", nvtx_color(8) );
			meso_atom->transfer_pre_comm();
        	nvtx_pop_range( "\u2198Comm" );

			if( force->pair->split_flag && pair_compute_flag ) {
        	    nvtx_push_range( "\u26EFForceB", nvtx_color(7) );
				force_clear( AtomAttribute::BULK );
				force->pair->compute_bulk( eflag, vflag );
				nvtx_pop_range( "\u26EFForceB" );
			}

			nvtx_push_range( "\u2198Comm", nvtx_color(8) );
			meso_device->event( "meso_atom::transfer_pre_comm" ).sync();
			nvtx_pop_range( "\u2198Comm" );
			timer->stamp();

			nvtx_push_range( "\u26D6Comm", nvtx_color(8) );
			comm->forward_comm();
			nvtx_pop_range( "\u26D6Comm" );

			nvtx_push_range( "\u2197Comm", nvtx_color(8) );
			meso_atom->transfer_post_comm();
			nvtx_pop_range( "\u2197Comm" );
			timer->stamp( TIME_COMM );

			if( force->pair->split_flag ) force_clear( AtomAttribute::BORDER );
        }

		// force computations
        nvtx_push_range( "Preforce", nvtx_color(9) );
        if( !force->pair->split_flag ) force_clear( AtomAttribute::LOCAL );
        if( n_pre_force ) modify->pre_force( vflag );
        nvtx_pop_range( "Preforce" );
        timer->stamp();
        check_error( __LINE__, __FILE__ );

		if( pair_compute_flag ) {
            if( force->pair->split_flag ) {
            	nvtx_push_range( "\u26EFForceH", nvtx_color(7) );
                force->pair->compute_border( eflag, vflag );
                nvtx_pop_range( "\u26EFForceH" );
            } else {
            	nvtx_push_range( "\u26EFForce", nvtx_color(7) );
                force->pair->compute( eflag, vflag );
                nvtx_pop_range( "\u26EFForceB" );
            }
            timer->stamp( TIME_PAIR );
        }
        check_error( __LINE__, __FILE__ );

		if( atom->molecular ) {
        	nvtx_push_range( "Bonded", nvtx_color(10) );
            if( force->bond ) force->bond->compute( eflag, vflag );
            if( force->angle ) force->angle->compute( eflag, vflag );
            if( force->dihedral ) force->dihedral->compute( eflag, vflag );
            if( force->improper ) force->improper->compute( eflag, vflag );
            nvtx_pop_range( "Bonded" );
            timer->stamp( TIME_BOND );
        }

		check_error( __LINE__, __FILE__ );

        // force modifications, final time integration, diagnostics
        nvtx_push_range( "FinalInt", nvtx_color(11) );

		if( n_post_force ) modify->post_force( vflag );

		modify->final_integrate();
        if( n_end_of_step ) modify->end_of_step();
        nvtx_pop_range( "FinalInt" );

        check_error( __LINE__, __FILE__ );

        // all output
        if( ntimestep == output->next ) {
            meso_atom->transfer_pre_output();
            timer->stamp();
            output->write( ntimestep );
            timer->stamp( TIME_OUTPUT );
        }

        meso_device->next_step();
    }

    meso_atom->transfer_pre_output();

    // stop the profiler
    meso_device->configure_profiler( update->ntimestep, update->nsteps );
}

/* ----------------------------------------------------------------------
     clear force on own & ghost atoms
     setup and clear other arrays as needed
------------------------------------------------------------------------- */

void ModifiedVerlet::force_clear( AtomAttribute::Descriptor range )
{
    meso_atom->meso_avec->force_clear( range, vflag );
}
