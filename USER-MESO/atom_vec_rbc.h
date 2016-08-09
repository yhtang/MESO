#ifndef LMP_MESO_ATOM_VEC_RBC_H
#define LMP_MESO_ATOM_VEC_RBC_H

#include "atom_vec.h"

namespace LAMMPS_NS {

class AtomVecRBC : public AtomVec {
public:
    AtomVecRBC( class LAMMPS * );
    virtual ~AtomVecRBC() {}
    void grow( int );
    void grow_reset();
    void copy( int, int, int );
    virtual int pack_comm( int, int *, double *, int, int * );
    virtual int pack_comm_vel( int, int *, double *, int, int * );
    virtual void unpack_comm( int, int, double * );
    virtual void unpack_comm_vel( int, int, double * );
    int pack_reverse( int, int, double * );
    void unpack_reverse( int, int *, double * );
    virtual int pack_border( int, int *, double *, int, int * );
    virtual int pack_border_vel( int, int *, double *, int, int * );
    int pack_border_hybrid( int, int *, double * );
    virtual void unpack_border( int, int, double * );
    virtual void unpack_border_vel( int, int, double * );
    int unpack_border_hybrid( int, int, double * );
    virtual int pack_exchange( int, double * );
    virtual int unpack_exchange( double * );
    int size_restart();
    int pack_restart( int, double * );
    int unpack_restart( double * );
    void create_atom( int, double * );
    void data_atom( double *, tagint, char ** );
    int data_atom_hybrid( int, char ** );
    void pack_data( double ** );
    int pack_data_hybrid( int, double * );
    void write_data( FILE *, int, double ** );
    int write_data_hybrid( FILE *, double * );
    bigint memory_usage();

protected:
    int * tag, *type, *mask;
    tagint * image;
    double ** x, ** v, ** f;
    int * molecule;
    int ** nspecial, ** special;
    int * num_bond;
    int ** bond_type, ** bond_atom;
    double ** bond_r0;
    int * num_angle;
    int ** angle_type;
    int ** angle_atom1, * * angle_atom2, * * angle_atom3;
    double ** angle_a0;
    int * num_dihedral;
    int ** dihedral_type;
    int ** dihedral_atom1, * * dihedral_atom2, * * dihedral_atom3, * * dihedral_atom4;
};

}

#endif
