#ifndef LMP_MESO_ATOM_VEC_SPH
#define LMP_MESO_ATOM_VEC_SPH

#include "atom_vec.h"

namespace LAMMPS_NS
{

class AtomVecSPH : public AtomVec
{
public:
    AtomVecSPH( class LAMMPS * );
    ~AtomVecSPH() {}
    void grow( int );
    void grow_reset();
    void copy( int, int, int );
    int pack_comm( int, int *, double *, int, int * );
    int pack_comm_vel( int, int *, double *, int, int * );
    void unpack_comm( int, int, double * );
    void unpack_comm_vel( int, int, double * );
    int pack_reverse( int, int, double * );
    void unpack_reverse( int, int *, double * );
    int pack_comm_hybrid( int, int *, double * );
    int unpack_comm_hybrid( int, int, double * );
    int pack_border_hybrid( int, int *, double * );
    int unpack_border_hybrid( int, int, double * );
    int pack_reverse_hybrid( int, int, double * );
    int unpack_reverse_hybrid( int, int *, double * );
    int pack_border( int, int *, double *, int, int * );
    int pack_border_vel( int, int *, double *, int, int * );
    void unpack_border( int, int, double * );
    void unpack_border_vel( int, int, double * );
    int pack_exchange( int, double * );
    int unpack_exchange( double * );
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

private:
    int *tag, *type, *mask;
    tagint *image;
    double **x, **v, **f;
    double *rho;
};

}

#endif
