#ifdef FIX_CLASS

FixStyle(boundary/fc/trp/special/meso,MesoFixBoundaryFcTRPSpecial)

#else

#ifndef LMP_MESO_FIX_BOUNDARY_FC_TRP_SPECIAL
#define LMP_MESO_FIX_BOUNDARY_FC_TRP_SPECIAL

#include "fix.h"
#include "meso.h"

// TRP = thermo-responsive polymer

namespace LAMMPS_NS {

class MesoFixBoundaryFcTRPSpecial : public Fix, protected MesoPointers {
public:
	MesoFixBoundaryFcTRPSpecial(class LAMMPS *, int, char **);
	~MesoFixBoundaryFcTRPSpecial();
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
	HostScalar<double> poly;
	class MesoPairEDPDTRPBase *pair;

	void prepare_coeff();
};

}

#endif

#endif
