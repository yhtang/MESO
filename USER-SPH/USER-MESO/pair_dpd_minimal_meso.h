#ifdef PAIR_CLASS

PairStyle(dpd/mini/meso,MesoPairDPDMini)

#else

#ifndef LMP_MESO_PAIR_DPD_MINI
#define LMP_MESO_PAIR_DPD_MINI

#include "pair_dpd_meso.h"

namespace LAMMPS_NS {

class MesoPairDPDMini : public Pair, protected MesoPointers {
 public:
	MesoPairDPDMini(class LAMMPS *);
	~MesoPairDPDMini();
	void   compute(int, int);
	void   compute_bulk(int, int);
	void   compute_border(int, int);
	void   settings(int, char **);
	void   coeff(int, char **);
	void   init_style();
	double init_one(int, int);
	void   write_restart(FILE *);
	void   read_restart(FILE *);
	void   write_restart_settings(FILE *);
	void   read_restart_settings(FILE *);

 protected:
	int      seed;

	// assuming cut = 1.0
	float   a0;
	float   gamma;
	float   sigma;

	class RanMars *random;

	virtual void allocate();
	virtual void compute_kernel(int, int, int, int);
	virtual uint seed_now();
};

}

#endif

#endif
