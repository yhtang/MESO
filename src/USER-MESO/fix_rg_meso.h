#ifdef FIX_CLASS

FixStyle(rg/meso,MesoFixRg)

#else

#ifndef LMP_MESO_FIX_RG
#define LMP_MESO_FIX_RG

#include "fix.h"
#include "pair.h"
#include "meso.h"

namespace LAMMPS_NS {

class MesoFixRg : public Fix, protected MesoPointers {
public:
	MesoFixRg(class LAMMPS *, int, char **);
	~MesoFixRg();
	virtual void init();
	virtual int setmask();
	virtual void setup(int);
	virtual void post_integrate();
	virtual double compute_scalar();

protected:
	std::string output;
	std::ofstream fout;
	int target_group;
	int n_mol;
	int n_smoothing, c_smoothing;
	double last_scalar;
	DeviceVector<r32> dev_com;
	DeviceScalar<r32> dev_com_mass;
	DeviceScalar<r32> dev_rg_sq;
	DeviceScalar<r64> dev_rg_sum;
	virtual void dump();
};

}

#endif

#endif
