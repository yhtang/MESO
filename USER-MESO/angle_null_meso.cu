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
#include "angle_null_meso.h"
#include "neighbor_meso.h"

using namespace LAMMPS_NS;

#define SMALL 0.000001

MesoAngleNull::MesoAngleNull( LAMMPS *lmp ):
    AngleHarmonic( lmp ),
    MesoPointers( lmp )
{
}

void MesoAngleNull::compute( int eflag, int vflag )
{
	printf("NO ANGLE\n");
}

