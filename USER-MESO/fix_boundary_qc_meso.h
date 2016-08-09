#ifdef FIX_CLASS

FixStyle(boundary/qc/meso,MesoFixBoundaryQc)

#else

#ifndef LMP_MESO_FIX_BOUNDARY_QC
#define LMP_MESO_FIX_BOUNDARY_QC

#include "fix.h"
#include "meso.h"

namespace LAMMPS_NS {

class MesoFixBoundaryQc : public Fix, protected MesoPointers {
public:
	MesoFixBoundaryQc(class LAMMPS *, int, char **);
	~MesoFixBoundaryQc();
	virtual int setmask();
	virtual void init();
	virtual void setup(int);
	virtual void post_force(int);

private:
	  double nx, ny, nz, H, cut;
	  double T0, a0;
	  HostScalar<double> poly;
};

}

#endif

#endif
