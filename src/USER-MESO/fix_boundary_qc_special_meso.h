#ifdef FIX_CLASS

FixStyle(boundary/qc/special/meso,MesoFixBoundaryQcSpecial)

#else

#ifndef LMP_MESO_FIX_BOUNDARY_QC_SPECIAL
#define LMP_MESO_FIX_BOUNDARY_QC_SPECIAL

#include "fix.h"
#include "meso.h"

namespace LAMMPS_NS {

class MesoFixBoundaryQcSpecial : public Fix, protected MesoPointers {
public:
	MesoFixBoundaryQcSpecial(class LAMMPS *, int, char **);
	~MesoFixBoundaryQcSpecial();
	virtual int setmask();
	virtual void init();
	virtual void setup(int);
	virtual void post_force(int);

private:
	double cx, cy, cz;
	double ox, oy, oz, cut;
	double radius, length;
	double T_H, T_C, a0;
	HostScalar<double> poly;
};

}

#endif

#endif
