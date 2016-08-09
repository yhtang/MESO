#ifndef LMP_MESO_DOMAIN_H
#define LMP_MESO_DOMAIN_H

#include "domain.h"
#include "meso.h"

namespace LAMMPS_NS
{

class MesoDomain : public Domain, protected MesoPointers
{
public:
    MesoDomain( class LAMMPS *lmp );

    virtual void pbc();
};

}

#endif
