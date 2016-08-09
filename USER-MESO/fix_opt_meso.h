#ifdef FIX_CLASS

FixStyle(opt/meso,FixOptMeso)

#else

#ifndef LMP_MESO_FIX_OPT
#define LMP_MESO_FIX_OPT

#include "fix_nve.h"
#include "meso.h"

namespace LAMMPS_NS {

class FixOptMeso : public Fix, protected MesoPointers
{
public:
	FixOptMeso(class LAMMPS *, int, char **);
	~FixOptMeso();
	virtual void init();
	virtual int  setmask();
	virtual void final_integrate();
protected:
	DeviceScalar<r64> dev_max_accl;
	r64  max_move;
	r64  noise_level;
};

}

#endif

#endif
