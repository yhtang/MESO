#ifdef FIX_CLASS

FixStyle(plane/meso,MesoFixPlane)

#else

#ifndef LMP_MESO_FIX_PLANE
#define LMP_MESO_FIX_PLANE

#include "fix.h"
#include "meso.h"

namespace LAMMPS_NS {

class MesoFixPlane : public Fix, protected MesoPointers {
public:
	MesoFixPlane(class LAMMPS *, int, char **);
	virtual int setmask();
	virtual void init();
	virtual void setup(int);
	virtual void post_force(int);
	virtual void post_integrate();
	virtual void pre_exchange();
	virtual void end_of_step();
protected:
	r64 nx, ny, nz, d, f;
	virtual void plane_force();
	virtual void bounce_back();
};

}

#endif

#endif
