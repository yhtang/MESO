#ifndef LMP_MESO_ERROR_H
#define LMP_MESO_ERROR_H

#include "pointers.h"
#include "meso.h"

namespace LAMMPS_NS
{

class MesoError : public Error, protected MesoPointers
{
public:
    MesoError( class LAMMPS *lmp ) : Error( lmp ), MesoPointers( lmp ) {}
};

}

#endif
