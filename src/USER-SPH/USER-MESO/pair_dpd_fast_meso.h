#ifdef PAIR_CLASS

PairStyle(dpd/fast/meso,MesoPairDPDFast)

#else

#ifndef LMP_MESO_PAIR_DPD_FAST
#define LMP_MESO_PAIR_DPD_FAST

#include "pair_dpd_meso.h"

namespace LAMMPS_NS {

class MesoPairDPDFast : public Pair, protected MesoPointers {
 public:
	MesoPairDPDFast(class LAMMPS *);
	~MesoPairDPDFast();
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
	bool     coeff_ready;
	DeviceScalar<r32> dev_coefficients;
	std::vector<r32> coeff_table;

	float   cut_global;
	float **cut;
	float **cut_inv;
	float **expw;
	float **a0;
	float **gamma;
	float **sigma;

	class RanMars *random;

	virtual void allocate();
	virtual void prepare_coeff();
	virtual void compute_kernel(int, int, int, int);
	virtual uint seed_now();
};

}

#endif

#endif
