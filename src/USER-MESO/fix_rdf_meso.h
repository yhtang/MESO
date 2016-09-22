#ifdef FIX_CLASS

FixStyle(rdf/meso,MesoFixRDF)

#else

#ifndef LMP_MESO_FIX_RDF
#define LMP_MESO_FIX_RDF

#include "fix.h"
#include "pair.h"
#include "meso.h"

namespace LAMMPS_NS {

class MesoFixRDF : public Fix, protected MesoPointers {
public:
	MesoFixRDF(class LAMMPS *, int, char **);
	~MesoFixRDF();
	virtual void init();
	virtual int setmask();
	virtual void setup(int);
	virtual void post_force(int);

protected:
	std::string output;
	int  n_hist, n_steps, n_every;
	int  j_group, j_groupbit;
	r32  rc;
	DeviceScalar<uint> dev_histogram;
	virtual void dump();
};

}

#endif

#endif
