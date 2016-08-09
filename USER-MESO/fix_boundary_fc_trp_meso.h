#ifdef FIX_CLASS

FixStyle(boundary/fc/trp/meso,MesoFixBoundaryFcTRP)

#else

#ifndef LMP_MESO_FIX_BOUNDARY_FC_TRP
#define LMP_MESO_FIX_BOUNDARY_FC_TRP

#include "fix.h"
#include "meso.h"

// TRP = thermo-responsive polymer

namespace LAMMPS_NS {

class MesoFixBoundaryFcTRP : public Fix, protected MesoPointers {
public:
	MesoFixBoundaryFcTRP(class LAMMPS *, int, char **);
	~MesoFixBoundaryFcTRP();
	virtual int setmask();
	virtual void init();
	virtual void setup(int);
	virtual void post_force(int);

private:
	int wall_type;
	double T0;
	double nx, ny, nz, H, cut;
	HostScalar<double> poly;
	class MesoPairEDPDTRPBase *pair;

	void prepare_coeff();
};

}

#endif

#endif
