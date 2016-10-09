#include "mpi.h"
#include "math.h"
#include "stdlib.h"
#include "string.h"
#include "limits.h"
#include "error.h"

#include "atom_meso.h"
#include "neighbor_meso.h"
#include "engine_meso.h"
#include "bin_meso.h"

using namespace LAMMPS_NS;

MesoBin::MesoBin( LAMMPS *lmp ) : Pointers( lmp ), MesoPointers( lmp ),
    dev_bin_id( lmp, "MesoBin::dev_bin_id" ),
    dev_atm_id( lmp, "MesoBin::dev_atm_id" ),
    dev_bin_location( lmp, "MesoBin::dev_bin_location" ),
    dev_bin_size( lmp, "MesoBin::dev_bin_size" ),
    dev_bin_size_local( lmp, "MesoBin::dev_bin_size_local" ),
    dev_bin_isghost( lmp, "MesoBin::dev_bin_isghost" ),
    tex_atm_id( lmp, "MesoBin::tex_atm_id" )
{
}

MesoBin::~MesoBin()
{
    tex_atm_id.unbind();
}

void MesoBin::alloc_bins()
{
    if( atom->nlocal + atom->nghost > dev_bin_id.n_elem() ) {
        int natom = ( atom->nlocal + atom->nghost ) * 1.1 ;
        dev_bin_id.grow( natom, false, false );
        dev_atm_id.grow( natom, false, false );
        tex_atm_id.bind( dev_atm_id );
    }
    if( neighbor->mbinx * neighbor->mbiny * neighbor->mbinz + 1 > dev_bin_size.n_elem() ) {
        int nbin = neighbor->mbinx * neighbor->mbiny * neighbor->mbinz + 1 ;
        dev_bin_location  .grow( nbin, false, false );
        dev_bin_isghost   .grow( nbin, false, false );
        dev_bin_size      .grow( nbin, false, false );
        dev_bin_size_local.grow( nbin, false, false );
    }

#ifdef LMP_MESO_LOG_L3
    fprintf( stderr, "<MESO> Bin space allocated: per-atom %d per-bin %d\n", dev_bin_id.n_elem(), dev_bin_size.n_elem() );
#endif
}





