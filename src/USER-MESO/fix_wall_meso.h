#ifdef FIX_CLASS

FixStyle(wall/meso,MesoFixWall)

#else

#ifndef LMP_MESO_FIX_WALL
#define LMP_MESO_FIX_WALL

#include "fix.h"
#include "meso.h"

namespace LAMMPS_NS {

class MesoFixWall : public Fix, protected MesoPointers {
public:
	MesoFixWall(class LAMMPS *, int, char **);
	virtual int setmask();
	virtual void init();
	virtual void setup(int);
	virtual void post_force(int);
	virtual void post_integrate();
	virtual void pre_exchange();
	virtual void end_of_step();
protected:
	bool x, y, z;
	r64 f, d;
	virtual void boundary_force();
	virtual void bounce_back();
};

}

#endif

#endif
