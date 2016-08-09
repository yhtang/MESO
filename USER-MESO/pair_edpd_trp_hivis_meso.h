#ifdef PAIR_CLASS

PairStyle(edpd/trp/hivis/meso,MesoPairEDPDTRPHiVis)

#else

#ifndef LMP_MESO_PAIR_EDPD_TRP_HIVIS
#define LMP_MESO_PAIR_EDPD_TRP_HIVIS

#include "pair_edpd_trp_base_meso.h"

namespace LAMMPS_NS {

class MesoPairEDPDTRPHiVis : public MesoPairEDPDTRPBase, protected MesoPointers {
 public:
	MesoPairEDPDTRPHiVis(class LAMMPS *);
	virtual ~MesoPairEDPDTRPHiVis();
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
	double **s;
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
