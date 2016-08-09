#ifdef FIX_CLASS

FixStyle(boundary/fc/meso,MesoFixBoundaryFc)

#else

#ifndef LMP_MESO_FIX_BOUNDARY_FC
#define LMP_MESO_FIX_BOUNDARY_FC

#include "fix.h"
#include "meso.h"

namespace LAMMPS_NS {

class MesoFixBoundaryFc : public Fix, protected MesoPointers {
public:
	MesoFixBoundaryFc(class LAMMPS *, int, char **);
	~MesoFixBoundaryFc();
	virtual int setmask();
	virtual void init();
	virtual void setup(int);
	virtual void post_force(int);

private:
	  double nx, ny, nz, H, cut;
	  HostScalar<double> poly;
};

}

#endif

#endif
