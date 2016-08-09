#ifdef PAIR_CLASS

PairStyle(edpd/meso,MesoPairEDPD)

#else

#ifndef LMP_MESO_PAIR_EDPD
#define LMP_MESO_PAIR_EDPD

#include "pair.h"
#include "meso.h"

namespace LAMMPS_NS {

namespace EDPD_COEFFICIENTS {
const static int n_coeff  = 10;
const static int p_cut    = 0;
const static int p_cutsq  = 1;
const static int p_cutinv = 2;
const static int p_expw   = 3;
const static int p_a0     = 4;
const static int p_gamma  = 5;
const static int p_sigma  = 6;
const static int p_cv     = 7;
const static int p_kappa  = 8;
const static int p_expw2  = 9;
}

class MesoPairEDPD : public Pair, protected MesoPointers {
 public:
	MesoPairEDPD(class LAMMPS *);
	~MesoPairEDPD();
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

	double   cut_global;
	double **cut;
	double **cut_inv;
	double **expw;
	double **a0;
	double **gamma;
	double **sigma;
	double **cv;
	double **kappa;
	double **expw2;

	class RanMars *random;

	virtual void allocate();
	virtual void prepare_coeff();
	virtual void compute_kernel(int, int, int, int);
	virtual uint seed_now();
};

}

#endif

#endif
