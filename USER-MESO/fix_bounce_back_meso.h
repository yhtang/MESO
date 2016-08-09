#ifdef FIX_CLASS

FixStyle(bounceback/meso,MesoFixBounceBack)

#else

#ifndef LMP_MESO_FIX_BOUNCE_BACK
#define LMP_MESO_FIX_BOUNCE_BACK

#include "fix.h"
#include "meso.h"

namespace LAMMPS_NS {

class MesoFixBounceBack : public Fix, protected MesoPointers {
public:
	MesoFixBounceBack(class LAMMPS *, int, char **);
	virtual int setmask();
	virtual void init();
	virtual void setup(int);
	virtual void pre_exchange();
	virtual void end_of_step();

protected:
	bool x, y, z;

	void bounce_back();
};

}

#endif

#endif
