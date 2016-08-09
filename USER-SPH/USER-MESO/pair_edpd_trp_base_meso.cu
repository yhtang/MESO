#include "mpi.h"
#include "lammps.h"
#include "pair.h"
#include "pair_edpd_trp_base_meso.h"

using namespace LAMMPS_NS;

MesoPairEDPDTRPBase::MesoPairEDPDTRPBase( LAMMPS *lmp ) :
		Pair( lmp ),
		coeff_ready( false ),
		dev_coefficients( lmp, "MesoPairEDPDTRPBase::dev_coefficients" )
{}
