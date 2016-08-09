#ifdef ATOM_CLASS

#else

#ifndef LMP_MESO_ATOM_VEC_DPD_MOLECULAR
#define LMP_MESO_ATOM_VEC_DPD_MOLECULAR

#include "meso.h"
#include "atom_vec_meso.h"

namespace LAMMPS_NS
{

class AtomVecDPDMolecular: public MesoAtomVec
{
public:
    AtomVecDPDMolecular( LAMMPS *lmp ) :
        MesoAtomVec( lmp ),
        dev_mole( lmp, "AtomVecDPDMolecular::dev_mole" ),
        dev_mole_pinned( lmp, "AtomVecDPDMolecular::dev_mole_pinned" ),
        dev_special_pinned( lmp, "AtomVecDPDMolecular::dev_special_pinned" ),
        dev_nexcl_full( lmp, "AtomVecDPDMolecular::dev_nexcl_full" ),
        dev_nexcl( lmp, "AtomVecDPDMolecular::dev_nexcl" ),
        dev_excl_table( lmp, "AtomVecDPDMolecular::dev_excl_table" )
    {
    }
    ~AtomVecDPDMolecular() {}

    virtual void grow_device( int );
    virtual void grow_exclusion();
    virtual void pin_host_array();
    virtual void unpin_host_array();

protected:
    DeviceScalar<int>  dev_mole;

    Pinned<int>        dev_mole_pinned;
    Pinned<int>        dev_special_pinned;

    DeviceScalar<int4> dev_nexcl;
    DeviceScalar<int>  dev_nexcl_full;
    DeviceScalar<int>  dev_excl_table; // table in column fasion

    virtual void transfer_impl( std::vector<CUDAEvent> &events, AtomAttribute::Descriptor per_atom_prop, TransferDirection direction, int p_beg, int n_atom, int p_stream, int p_inc, int* permute_to, int* permute_from, int action, bool streamed );
    CUDAEvent transfer_exclusion( TransferDirection direction, int* permute_from, int p_beg, int n_atom, CUDAStream stream, int action );
};

}

#endif

#endif
