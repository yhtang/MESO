#ifndef LMP_MESO_COMM
#define LMP_MESO_COMM

#include "comm.h"
#include "meso.h"
#include "prefix_meso.h"

namespace LAMMPS_NS
{

class MesoComm: public Comm, protected MesoPointers
{
public:
    MesoComm( class LAMMPS *lmp ) : Comm( lmp ), MesoPointers( lmp )
    {
        ghost_velocity = 1;
    }
    virtual ~MesoComm() {}

    virtual void borders();
    virtual void borderness( std::vector<int> & );
    virtual void exchange();

protected:
    template<class PRED>
    void pick_border( int nfirst, int nlast, int iswap, int& nsend, PRED pred, size_t ntd )
    {
        static std::vector<std::vector<int> > send;
        if( send.size() < ntd ) send.resize( ntd );

        if( OMPDEBUG ) printf( "%d %s\n", __LINE__, __FILE__ );
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            if( tid < ntd ) {
                std::vector<int> &mysend = send[tid];
                mysend.clear();

                // manual for loop to ensure locality preservation
                int2 work = split_work( nfirst, nlast, tid, ntd );
                for( int i = work.x; i < work.y; i++ ) {
                    if( pred( i ) ) {
                        mysend.push_back( i );
                    }
                }
            }
        }
        nsend = 0;
        for( int i = 0 ; i < ntd ; i++ ) {
            nsend += send[i].size();
        }
        if( nsend >= maxsendlist[iswap] ) grow_list( iswap, nsend );

        int p = 0;
        for( int i = 0 ; i < ntd ; i++ ) {
            std::vector<int> &s = send[i];
            for( int j = 0 ; j < s.size() ; j++ ) {
                sendlist[iswap][p++] = s[j];
            }
        }
    }
    template<class PRED>
    void get_borderness( std::vector<int> &borderness, int nfirst, int nlast, PRED pred, int ntd )
    {
        int tid = omp_get_thread_num();
        if( tid < ntd ) {
            int2 work = split_work( nfirst, nlast, tid, ntd );
            for( int i = work.x; i < work.y; i++ ) {
                if( pred( i ) ) borderness[i]++;
            }
        }
    }
};

// predication functor for border determination
struct pred_border_single {
public:
    pred_border_single( int _dim, double _lo, double _hi, double **_x ) : dim( _dim ), lo( _lo ), hi( _hi ), x( _x ) {}
    inline bool operator()( int i ) const
    {
        return x[i][dim] >= lo && x[i][dim] <= hi;
    }
protected:
    int dim;
    double lo, hi;
    double** x;
};

struct pred_border_multi {
public:
    pred_border_multi( int _dim, int *_type, double *_mlo, double *_mhi, double **_x ) : dim( _dim ), type( _type ), mlo( _mlo ), mhi( _mhi ), x( _x ) {}
    inline bool operator()( int i ) const
    {
        return x[i][dim] >= mlo[type[i]] && x[i][dim] <= mhi[type[i]];
    }
protected:
    int dim;
    int *type;
    double *mlo, *mhi;
    double** x;
};

}

#endif
