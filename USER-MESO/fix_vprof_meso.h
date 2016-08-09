#ifdef FIX_CLASS

FixStyle(vprof/meso,MesoFixVProf)

#else

#ifndef LMP_MESO_FIX_VPROF
#define LMP_MESO_FIX_VPROF

#include "fix.h"

namespace LAMMPS_NS {

class MesoFixVProf : public Fix, protected MesoPointers {
public:
	MesoFixVProf(class LAMMPS *, int, char **);
	~MesoFixVProf();
	virtual int setmask();
	virtual void setup(int);
	virtual void post_integrate();

protected:
	std::string output;
	int  n_bin, along, component;
	r64  bin_size, every, window;
	bigint last_dump_time;
	DeviceScalar<r64>  dev_histogram;
	DeviceScalar<uint> dev_count;
	virtual void dump( bigint tstamp );

	void compute();
};

}

#endif

#endif
