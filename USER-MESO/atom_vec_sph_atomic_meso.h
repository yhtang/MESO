#ifdef ATOM_CLASS

AtomStyle(sph/atomic/meso,AtomVecSPHAtomic)

#else

#ifndef LMP_MESO_ATOM_VEC_SPH_ATOMIC
#define LMP_MESO_ATOM_VEC_SPH_ATOMIC

#include "atom_vec_sph.h"
#include "atom_vec_meso.h"

namespace LAMMPS_NS {

class AtomVecSPHAtomic: public AtomVecSPH, public MesoAtomVec
{
public:
	AtomVecSPHAtomic(LAMMPS *lmp) :
		AtomVecSPH(lmp),
		MesoAtomVec(lmp),
		dev_rho       ( lmp, "AtomVecSPHAtomic::dev_rho" ),
		dev_rho_pinned( lmp, "AtomVecSPHAtomic::dev_rho_pinned" )
	{
		cudable        = 1;
		comm_x_only    = 0;

		pre_sort     = AtomAttribute::LOCAL  | AtomAttribute::COORD;
		post_sort    = AtomAttribute::LOCAL  | AtomAttribute::ESSENTIAL;
		pre_border   = AtomAttribute::BORDER | AtomAttribute::ESSENTIAL;
		post_border  = AtomAttribute::GHOST  | AtomAttribute::ESSENTIAL;
		pre_comm     = AtomAttribute::BORDER | AtomAttribute::COORD | AtomAttribute::VELOC;
		post_comm    = AtomAttribute::GHOST  | AtomAttribute::COORD | AtomAttribute::VELOC;
		pre_exchange = AtomAttribute::LOCAL  | AtomAttribute::ESSENTIAL;
		pre_output   = AtomAttribute::LOCAL  | AtomAttribute::ESSENTIAL | AtomAttribute::FORCE | AtomAttribute::RHO;
	}
	~AtomVecSPHAtomic() {}

	virtual void grow(int);
	virtual void grow_cpu(int);
	virtual void grow_device( int nmax_new );
	virtual void data_atom_target(int, double*, int, char**);
	virtual void pin_host_array();
	virtual void unpin_host_array();
	virtual void dp2sp_merged( int seed, int p_beg, int p_end, bool offset = false );
	virtual void force_clear( AtomAttribute::Descriptor, int );

protected:
	DeviceScalar<r64> dev_rho;
	Pinned<r64> dev_rho_pinned;

	virtual void transfer_impl( std::vector<CUDAEvent> &events, AtomAttribute::Descriptor per_atom_prop, TransferDirection direction, int p_beg, int n_atom, int p_stream, int p_inc, int* permute_to, int* permute_from, int action, bool streamed);
};

}

#endif

#endif
