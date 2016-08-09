#ifdef FIX_CLASS

FixStyle(vx2T/edpd/meso,FixEDPDVx2TMeso)

#else

#ifndef LMP_MESO_FIX_VX2T
#define LMP_MESO_FIX_VX2T

#include "fix_nve.h"
#include "meso.h"

namespace LAMMPS_NS {

// for use with read_dump/rerun
// since we have to encode T as vx in dump/edpd file
class FixEDPDVx2TMeso : public Fix, protected MesoPointers
{
public:
	FixEDPDVx2TMeso(class LAMMPS *, int, char **);
	virtual int setmask();
	virtual void setup_pre_neighbor();
};

}

#endif

#endif
