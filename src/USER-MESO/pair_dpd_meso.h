#ifdef PAIR_CLASS

PairStyle(dpd/meso,MesoPairDPD)

#else

#ifndef LMP_MESO_PAIR
#define LMP_MESO_PAIR

#include "pair.h"
#include "meso.h"

namespace LAMMPS_NS {

namespace DPD_COEFFICIENTS {
const static int n_coeff  = 7;
const static int p_cut    = 0;
const static int p_cutsq  = 1;
const static int p_cutinv = 2;
const static int p_expw   = 3;
const static int p_a0     = 4;
const static int p_gamma  = 5;
const static int p_sigma  = 6;
}

class MesoPairDPD : public Pair, protected MesoPointers {
 public:
	MesoPairDPD(class LAMMPS *);
	~MesoPairDPD();
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
	double single(int, int, int, int, double, double, double, double &);

 protected:
	int      seed;
	bool     coeff_ready;
	DeviceScalar<r64> dev_coefficients;
	std::vector<r64>       coeff_table;

	double   cut_global;
	double **cut;
	double **cut_inv;
	double **expw;
	double **a0;
	double **gamma;
	double **sigma;


	class RanMars *random;

	virtual void allocate();
	virtual void prepare_coeff();
	virtual void compute_kernel(int, int, int, int);
	virtual uint seed_now();
};

}

#endif

#endif
