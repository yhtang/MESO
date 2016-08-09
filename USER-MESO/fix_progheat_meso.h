#ifdef FIX_CLASS

FixStyle(progheat/meso,MesoFixProgHeat)

#else

#ifndef LMP_MESO_FIX_PROGHEAT
#define LMP_MESO_FIX_PROGHEAT

#include "fix.h"

namespace LAMMPS_NS {

class MesoFixProgHeat : public Fix, protected MesoPointers {
public:
	MesoFixProgHeat(class LAMMPS *, int, char **);
	~MesoFixProgHeat();
	virtual int setmask();
	virtual void init();
	virtual void setup(int);
	virtual void post_force(int);
	virtual double compute_scalar();

private:
	r64 delta, Cv;
	std::string T_template; // T_expr = f(t)

	char **T_command;
    std::string t_user, T_varname;
    HostScalar<r64> hst_meanT;
    DeviceScalar<r32> dev_dQ;
};

}

#endif

#endif
