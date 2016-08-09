#ifdef PAIR_CLASS

PairStyle(pnipam/meso,MesoPairEDPDTRP)
PairStyle(edpd/trp/meso,MesoPairEDPDTRP)

#else

#ifndef LMP_MESO_PAIR_EDPD_TRP
#define LMP_MESO_PAIR_EDPD_TRP

#include "pair_edpd_trp_base_meso.h"

namespace LAMMPS_NS {

class MesoPairEDPDTRP : public MesoPairEDPDTRPBase, protected MesoPointers {
 public:
	MesoPairEDPDTRP(class LAMMPS *);
	virtual ~MesoPairEDPDTRP();
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

	double   cut_global;
	double **cut;
	double **cut_inv;
	double **a0;
	double **gamma;
	double **cv;
	double **kappa;
	double **theta; // theta-temperature: temperature of chi=0
	double **da;    // a_ij = a_ij_0 + da / ( 1 + exp( omega * ( T - theta ) ) )
	double **omega;

	class RanMars *random;

	virtual void allocate();
	virtual void prepare_coeff();
	virtual void compute_kernel(int, int, int, int);
	virtual uint seed_now();
};

}

#endif

#endif
