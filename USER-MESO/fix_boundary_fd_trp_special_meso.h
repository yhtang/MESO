#ifdef FIX_CLASS

FixStyle(boundary/fd/trp/special/meso,MesoFixBoundaryFdTRPSpecial)

#else

#ifndef LMP_MESO_FIX_BOUNDARY_FD_TRP_SPECIAL
#define LMP_MESO_FIX_BOUNDARY_FD_TRP_SPECIAL

#include "fix.h"
#include "meso.h"

// TRP = thermo-responsive polymer

namespace LAMMPS_NS {

class MesoFixBoundaryFdTRPSpecial : public Fix, protected MesoPointers {
public:
	MesoFixBoundaryFdTRPSpecial(class LAMMPS *, int, char **);
	~MesoFixBoundaryFdTRPSpecial();
	virtual int setmask();
	virtual void init();
	virtual void setup(int);
	virtual void post_force(int);

private:
	int wall_type;
	double cx, cy, cz;
	double ox, oy, oz, cut;
	double radius, length;
	double T_H, T_C;
	// poly + R:
	// coefficient for the functional form: A0/h + polynomial(h)
	double A0;
	HostScalar<double> poly;
	class MesoPairEDPDTRPBase *pair;

	void prepare_coeff();
};

}

#endif

#endif
