#ifdef FIX_CLASS

FixStyle(addheat/meso,MesoFixAddHeat)

#else

#ifndef LMP_MESO_FIX_ADDHEAT
#define LMP_MESO_FIX_ADDHEAT

#include "fix.h"

namespace LAMMPS_NS {

class MesoFixAddHeat : public Fix, protected MesoPointers {
public:
	MesoFixAddHeat(class LAMMPS *, int, char **);
	virtual int setmask();
	virtual void init();
	virtual void setup(int);
	virtual void post_force(int);

private:
	r64 heat;
};

}

#endif

#endif
