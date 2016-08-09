#ifdef FIX_CLASS

FixStyle(addforce/meso,MesoFixAddForce)

#else

#ifndef LMP_MESO_FIX_ADD_FORCE
#define LMP_MESO_FIX_ADD_FORCE

#include "fix.h"

namespace LAMMPS_NS {

class MesoFixAddForce : public Fix, protected MesoPointers {
public:
	MesoFixAddForce(class LAMMPS *, int, char **);
	virtual int setmask();
	virtual void init();
	virtual void setup(int);
	virtual void post_force(int);

private:
	r64 fx, fy, fz;
};

}

#endif

#endif
