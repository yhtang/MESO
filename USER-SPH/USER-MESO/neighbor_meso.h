#ifndef LMP_MESO_NEIGHBOR
#define LMP_MESO_NEIGHBOR

#include "neighbor.h"
#include "meso.h"
#include "sort_meso.h"
#include "bin_meso.h"


namespace LAMMPS_NS
{

struct BoxDim {
    double3 hi, lo;
    BoxDim() {}
    BoxDim( double3 nhi, double3 nlo )
    {
        hi = nhi, lo = nlo ;
    }
    BoxDim& operator = ( BoxDim &x )
    {
        hi = x.hi, lo = x.lo;
        return *this;
    }
};

class MesoNeighbor : public Neighbor, protected MesoPointers
{
public:
    MesoNeighbor( class LAMMPS * );
    ~MesoNeighbor();

    virtual void init();
    virtual void setup_bins();
    int mbins() const { return mbinx * mbiny * mbinz; }

    BoxDim  my_box;
    MesoBin cuda_bin;

    HostScalar<int> max_local_bin_size;
    r64    local_particle_density;
    r64    expected_neigh_count;
    r64    expected_bin_size;

    typedef std::map<int, class MesoNeighList *> dlist_container;
    typedef std::map<int, class MesoNeighList *>::iterator dlist_iter;
    dlist_container lists_device;

protected:
    SortPlan<16, uint, int> sorter;

    virtual void choose_build( int, class NeighRequest * );
    virtual void choose_stencil( int, class NeighRequest * );

    virtual void full_bin_meso( class NeighList * );
    virtual void full_bin_meso_ghost( class NeighList * );
    virtual void stencil_full_bin_3d_meso( class NeighList *, int, int, int );

    virtual void binning_meso( class MesoNeighList *list, bool ghost );
    virtual void filter_exclusion_meso( class MesoNeighList * );

    virtual void bond_all();
    virtual void bond_partial();

    virtual void angle_all();
    virtual void angle_partial();

	virtual void dihedral_all();
	virtual void dihedral_partial();

//  virtual void improper_all() {Neighbor::improper_all();}
//  virtual void improper_partial() {Neighbor::improper_partial();}

};

}

#endif
