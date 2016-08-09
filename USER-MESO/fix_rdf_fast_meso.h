#ifdef FIX_CLASS

FixStyle(rdf/fast/meso,MesoFixRDFFast)

#else

#ifndef LMP_MESO_FIX_RDF_FAST
#define LMP_MESO_FIX_RDF_FAST

#include "fix.h"
#include "pair.h"
#include "meso.h"

namespace LAMMPS_NS {

class MesoFixRDFFast : public Fix, protected MesoPointers {
public:
	MesoFixRDFFast(class LAMMPS *, int, char **);
	~MesoFixRDFFast();
	virtual void init();
	virtual int setmask();
	virtual void setup(int);
	virtual void post_force(int);

protected:
	std::string output;
	int  n_bin, n_steps, n_every;
	int  j_group, j_groupbit;
	r32  rc;
	DeviceScalar<uint> dev_histogram;
	virtual void dump();
};

}

#endif

#endif
