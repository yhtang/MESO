#include "mpi.h"
#include "string.h"
#include "atom.h"
#include "update.h"
#include "force.h"
#include "pair.h"
#include "bond.h"
#include "angle.h"
#include "dihedral.h"
#include "improper.h"
#include "kspace.h"
#include "modify.h"
#include "domain.h"
#include "error.h"

#include "atom_meso.h"
#include "engine_meso.h"
#include "atom_vec_meso.h"
#include "compute_pe_meso.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

MesoComputePE::MesoComputePE( LAMMPS *lmp, int narg, char **arg ) :
    Compute( lmp, narg, arg ),
    MesoPointers( lmp ),
    per_atom_energy( lmp, "MesoComputePE::per_atom_energy" )

{
    if( narg < 3 ) error->all( FLERR, "Illegal compute pe command" );
    if( igroup ) error->all( FLERR, "Compute pe must use group all" );

    // settings
    scalar_flag = 1;
    extscalar = 1;
    peflag = 1;
    timeflag = 1;

    if( narg == 3 ) {
        pairflag = 1;
        bondflag = angleflag = dihedralflag = improperflag = 1;
        thermoflag = 1;
    } else {
        pairflag = 0;
        bondflag = angleflag = dihedralflag = improperflag = 0;
        thermoflag = 0;
        int iarg = 3;
        while( iarg < narg ) {
            if( strcmp( arg[iarg], "pair" ) == 0 ) pairflag = 1;
            else if( strcmp( arg[iarg], "bond" ) == 0 ) bondflag = 1;
            else if( strcmp( arg[iarg], "angle" ) == 0 ) angleflag = 1;
            else if( strcmp( arg[iarg], "dihedral" ) == 0 ) dihedralflag = 1;
            else if( strcmp( arg[iarg], "improper" ) == 0 ) improperflag = 1;
            else error->all( FLERR, "Illegal compute pe command" );
            iarg++;
        }
    }

    // allocated pinned memory
    per_atom_energy.grow( 5 );
}

/* ---------------------------------------------------------------------- */

double MesoComputePE::compute_scalar()
{
    invoked_scalar = update->ntimestep;
    if( update->eflag_global != invoked_scalar )
        error->all( FLERR, "Energy was not tallied on needed timestep" );

    double one = 0.0;
    memset( per_atom_energy, 0, per_atom_energy.n_byte() );

    if( pairflag && force->pair ) {
        size_t threads_per_block = 1024 ;
        gpu_reduce_sum_host<r64> <<< 1, threads_per_block, 0, meso_device->stream() >>> (
            meso_atom->dev_e_pair,
            per_atom_energy.ptr() + 0,
            atom->nlocal );
    }

    if( atom->molecular ) {
        size_t threads_per_block = 1024 ;
        if( bondflag && force->bond )
            gpu_reduce_sum_host<r64> <<< 1, threads_per_block, 0, meso_device->stream() >>> (
                meso_atom->dev_e_bond,
                per_atom_energy.ptr() + 1,
                atom->nlocal );
        if( angleflag && force->angle )
            gpu_reduce_sum_host<r64> <<< 1, threads_per_block, 0, meso_device->stream() >>> (
                meso_atom->dev_e_angle,
                per_atom_energy.ptr() + 2,
                atom->nlocal );
        if( dihedralflag && force->dihedral )
            gpu_reduce_sum_host<r64> <<< 1, threads_per_block, 0, meso_device->stream() >>> (
                meso_atom->dev_e_dihed,
                per_atom_energy.ptr() + 3,
                atom->nlocal );
        if( improperflag && force->improper )
            gpu_reduce_sum_host<r64> <<< 1, threads_per_block, 0, meso_device->stream() >>> (
                meso_atom->dev_e_impro,
                per_atom_energy.ptr() + 4,
                atom->nlocal );
    }
    meso_device->stream().sync();

    if( pairflag     && force->pair )     force -> pair     -> eng_vdwl = per_atom_energy[0];
    if( atom->molecular && bondflag     && force->bond )     force -> bond     -> energy = per_atom_energy[1];
    if( atom->molecular && angleflag    && force->angle )    force -> angle    -> energy = per_atom_energy[2];
    if( atom->molecular && dihedralflag && force->dihedral ) force -> dihedral -> energy = per_atom_energy[3];
    if( atom->molecular && improperflag && force->improper ) force -> improper -> energy = per_atom_energy[4];
    one += per_atom_energy[0] + per_atom_energy[1] + per_atom_energy[2] + per_atom_energy[3] + per_atom_energy[4];

    MPI_Allreduce( &one, &scalar, 1, MPI_DOUBLE, MPI_SUM, world );

    if( pairflag && force->pair && force->pair->tail_flag ) {
        double volume = domain->xprd * domain->yprd * domain->zprd;
        scalar += force->pair->etail / volume;
    }

    if( thermoflag && modify->n_thermo_energy ) scalar += modify->thermo_energy();

    return scalar;
}
