#include "mpi.h"
#include "memory.h"
#include "lammps.h"

#include "engine_meso.h"
#include "timer_meso.h"

using namespace LAMMPS_NS;

MesoTimer::MesoTimer( class LAMMPS *lmp ) : Timer( lmp ), MesoPointers( lmp )
{
}

CUDAEvent MesoTimer::device_stamp( std::string tag )
{
    CUDAEvent e = meso_device->event( tag );
    e.record( meso_device->stream() );
    return e;
}
