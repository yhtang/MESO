#ifndef LMP_MESO_TIMER_H
#define LMP_MESO_TIMER_H

#include "timer.h"
#include "meso.h"

namespace LAMMPS_NS
{

class MesoTimer : public Timer, protected MesoPointers
{
public:
    MesoTimer( class LAMMPS *lmp );

    CUDAEvent device_stamp( std::string tag );
};

}

#endif
