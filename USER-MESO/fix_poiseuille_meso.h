#ifdef FIX_CLASS

FixStyle(pois/meso,MesoFixPoiseuille)

#else

#ifndef LMP_MESO_FIX_POISEUILLE
#define LMP_MESO_FIX_POISEUILLE

#include "fix.h"

namespace LAMMPS_NS {

class MesoFixPoiseuille : public Fix, protected MesoPointers {
public:
	MesoFixPoiseuille(class LAMMPS *, int, char **);
	virtual int setmask();
	virtual void init();
	virtual void setup(int);
	virtual void post_force(int);

private:
	int dim_ortho, dim_force;
	r64 strength, bisect_frac;
};

}

#endif

#endif
