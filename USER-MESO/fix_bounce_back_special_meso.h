#ifdef FIX_CLASS

FixStyle(bounceback/special/meso,MesoFixBounceBackSpecial)

#else

#ifndef LMP_MESO_FIX_BOUNCE_BACK_SPECIAL
#define LMP_MESO_FIX_BOUNCE_BACK_SPECIAL

#include "fix.h"
#include "meso.h"

namespace LAMMPS_NS {

class MesoFixBounceBackSpecial : public Fix, protected MesoPointers {
public:
	MesoFixBounceBackSpecial(class LAMMPS *, int, char **);
	virtual int setmask();
	virtual void init();
	virtual void setup(int);
	virtual void pre_exchange();
	virtual void end_of_step();

protected:
	double cx, cy, cz;
	double ox, oy, oz;
	double radius;

	void bounce_back();
};

}

#endif

#endif
