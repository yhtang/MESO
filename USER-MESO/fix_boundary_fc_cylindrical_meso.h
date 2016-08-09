#ifdef FIX_CLASS

FixStyle(boundary/fc/cylindrical/meso,MesoFixBoundaryFcCylindrical)

#else

#ifndef LMP_MESO_FIX_BOUNDARY_FC_CYLINDRICAL
#define LMP_MESO_FIX_BOUNDARY_FC_CYLINDRICAL

#include "fix.h"
#include "meso.h"

namespace LAMMPS_NS {

class MesoFixBoundaryFcCylindrical : public Fix, protected MesoPointers {
public:
	MesoFixBoundaryFcCylindrical(class LAMMPS *, int, char **);
	~MesoFixBoundaryFcCylindrical();
	virtual int setmask();
	virtual void init();
	virtual void setup(int);
	virtual void post_force(int);

private:
	double cx, cy, cz;
	double ox, oy, oz, cut, a0;
	double radius, length;
	HostScalar<double> poly;

	void prepare_coeff();
};

}

#endif

#endif
