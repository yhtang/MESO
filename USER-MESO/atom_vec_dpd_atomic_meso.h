#ifdef ATOM_CLASS

AtomStyle(dpd/atomic/meso,AtomVecDPDAtomic)

#else

#ifndef LMP_MESO_ATOM_VEC_DPD_ATOMIC
#define LMP_MESO_ATOM_VEC_DPD_ATOMIC

#include "atom_vec_atomic.h"
#include "atom_vec_meso.h"

namespace LAMMPS_NS {

class AtomVecDPDAtomic: public AtomVecAtomic, public MesoAtomVec
{
public:
	AtomVecDPDAtomic(LAMMPS *lmp) :
		AtomVecAtomic(lmp),
		MesoAtomVec(lmp)
	{
		cudable = 1;
		comm_x_only = 0;

		pre_sort     = AtomAttribute::LOCAL  | AtomAttribute::COORD;
		post_sort    = AtomAttribute::LOCAL  | AtomAttribute::ESSENTIAL;
		pre_border   = AtomAttribute::BORDER | AtomAttribute::ESSENTIAL;
		post_border  = AtomAttribute::GHOST  | AtomAttribute::ESSENTIAL;
		pre_comm     = AtomAttribute::BORDER | AtomAttribute::COORD | AtomAttribute::VELOC;
		post_comm    = AtomAttribute::GHOST  | AtomAttribute::COORD | AtomAttribute::VELOC;
		pre_exchange = AtomAttribute::LOCAL  | AtomAttribute::ESSENTIAL;
		pre_output   = AtomAttribute::LOCAL  | AtomAttribute::ESSENTIAL | AtomAttribute::FORCE;
	}
	~AtomVecDPDAtomic() {}

	virtual void grow(int);
	virtual void grow_cpu(int);
	virtual void grow_device( int nmax_new );
	virtual void data_atom_target(int, double*, int, char**);

	virtual int pack_border_vel(int, int *, double *, int, int *);
	virtual void unpack_border_vel(int, int, double *);
	virtual int pack_comm_vel(int, int *, double *, int, int *);
	virtual void unpack_comm_vel(int, int, double *);
};

}

#endif

#endif
