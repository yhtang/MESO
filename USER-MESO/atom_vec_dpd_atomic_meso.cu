#include "stdlib.h"
#include "domain.h"
#include "modify.h"
#include "fix.h"
#include "memory.h"
#include "error.h"
#include "bond.h"
#include "force.h"
#include "update.h"

#include "meso.h"
#include "atom_meso.h"
#include "comm.h"
#include "engine_meso.h"
#include "neighbor_meso.h"
#include "atom_vec_dpd_atomic_meso.h"

using namespace LAMMPS_NS;

void AtomVecDPDAtomic::grow( int n )
{
    unpin_host_array();
    if( n == 0 ) n = max( nmax + growth_inc, ( int )( nmax * growth_mul ) );
    grow_cpu( n );
    grow_device( n );
    pin_host_array();
}

void AtomVecDPDAtomic::grow_cpu( int n )
{
    AtomVecAtomic::grow( n );
}

void AtomVecDPDAtomic::grow_device( int nmax_new )
{
    MesoAtomVec::grow_device( nmax_new );
}

void AtomVecDPDAtomic::data_atom_target( int i, double *coord, int imagetmp, char **values )
{
    tag[i] = atoi( values[0] );
    if( tag[i] <= 0 )
        error->one( __FILE__, __LINE__, "Invalid atom ID in Atoms section of data file" );

    type[i] = atoi( values[1] );
    if( type[i] <= 0 || type[i] > atom->ntypes )
        error->one( __FILE__, __LINE__, "Invalid atom type in Atoms section of data file" );

    x[i][0] = coord[0];
    x[i][1] = coord[1];
    x[i][2] = coord[2];

    image[i] = imagetmp;

    mask[i] = 1;
    v[i][0] = 0.0;
    v[i][1] = 0.0;
    v[i][2] = 0.0;
}

int AtomVecDPDAtomic::pack_border_vel( int n, int *list, double *buf, int pbc_flag, int *pbc )
{
    int sz = size_border + size_velocity;
    if( pbc_flag == 0 ) {
        for( int i = 0; i < n; i++ ) {
            int j = list[i];
            int m = i * sz;
            buf[m++] = x[j][0];
            buf[m++] = x[j][1];
            buf[m++] = x[j][2];
            pack( buf + m++, type[j] );
            pack( buf + m++, mask[j] );
            pack( buf + m++, tag[j] );
            buf[m++] = v[j][0];
            buf[m++] = v[j][1];
            buf[m++] = v[j][2];
        }
    } else {
        double dx, dy, dz;
        if( domain->triclinic == 0 ) {
            dx = pbc[0] * domain->xprd;
            dy = pbc[1] * domain->yprd;
            dz = pbc[2] * domain->zprd;
        } else {
            dx = pbc[0];
            dy = pbc[1];
            dz = pbc[2];
        }
        if( !deform_vremap ) {
            for( int i = 0; i < n; i++ ) {
                int j = list[i];
                int m = i * sz;
                buf[m++] = x[j][0] + dx;
                buf[m++] = x[j][1] + dy;
                buf[m++] = x[j][2] + dz;
                pack( buf + m++, type[j] );
                pack( buf + m++, mask[j] );
                pack( buf + m++, tag[j] );
                buf[m++] = v[j][0];
                buf[m++] = v[j][1];
                buf[m++] = v[j][2];
            }
        } else {
            double dvx = pbc[0] * h_rate[0] + pbc[5] * h_rate[5] + pbc[4] * h_rate[4];
            double dvy = pbc[1] * h_rate[1] + pbc[3] * h_rate[3];
            double dvz = pbc[2] * h_rate[2];
            for( int i = 0; i < n; i++ ) {
                int j = list[i];
                int m = i * sz;
                buf[m++] = x[j][0] + dx;
                buf[m++] = x[j][1] + dy;
                buf[m++] = x[j][2] + dz;
                pack( buf + m++, type[j] );
                pack( buf + m++, mask[j] );
                pack( buf + m++, tag[j] );
                if( mask[i] & deform_groupbit ) {
                    buf[m++] = v[j][0] + dvx;
                    buf[m++] = v[j][1] + dvy;
                    buf[m++] = v[j][2] + dvz;
                } else {
                    buf[m++] = v[j][0];
                    buf[m++] = v[j][1];
                    buf[m++] = v[j][2];
                }
            }
        }
    }

    int m = n * sz;
    if( atom->nextra_border )
        for( int iextra = 0; iextra < atom->nextra_border; iextra++ )
            m += modify->fix[atom->extra_border[iextra]]->pack_border( n, list, &buf[m] );

    return m;
}

void AtomVecDPDAtomic::unpack_border_vel( int n, int first, double *buf )
{
    int last;
    int sz = size_border + size_velocity;

    last = first + n;
    while( last > nmax ) grow( 0 );

    for( int i = first; i < last; i++ ) {
        int m = ( i - first ) * sz;
        x[i][0] = buf[m++];
        x[i][1] = buf[m++];
        x[i][2] = buf[m++];
        unpack( buf + m++, type[i] );
        unpack( buf + m++, mask[i] );
        unpack( buf + m++, tag[i] );
        v[i][0] = buf[m++];
        v[i][1] = buf[m++];
        v[i][2] = buf[m++];
    }

    int m = n * sz;
    if( atom->nextra_border )
        for( int iextra = 0; iextra < atom->nextra_border; iextra++ )
            m += modify->fix[atom->extra_border[iextra]]->
                 unpack_border( n, first, &buf[m] );
}

int AtomVecDPDAtomic::pack_comm_vel( int n, int *list, double *buf,
                                     int pbc_flag, int *pbc )
{
    int sz = size_forward + size_velocity;

    if( pbc_flag == 0 ) {
        for( int i = 0; i < n; i++ ) {
            int j = list[i];
            int m = i * sz;
            buf[m++] = x[j][0];
            buf[m++] = x[j][1];
            buf[m++] = x[j][2];
            buf[m++] = v[j][0];
            buf[m++] = v[j][1];
            buf[m++] = v[j][2];
        }
    } else {
        double dx, dy, dz;
        if( domain->triclinic == 0 ) {
            dx = pbc[0] * domain->xprd;
            dy = pbc[1] * domain->yprd;
            dz = pbc[2] * domain->zprd;
        } else {
            dx = pbc[0] * domain->xprd + pbc[5] * domain->xy + pbc[4] * domain->xz;
            dy = pbc[1] * domain->yprd + pbc[3] * domain->yz;
            dz = pbc[2] * domain->zprd;
        }
        if( !deform_vremap ) {
            for( int i = 0; i < n; i++ ) {
                int j = list[i];
                int m = i * sz;
                buf[m++] = x[j][0] + dx;
                buf[m++] = x[j][1] + dy;
                buf[m++] = x[j][2] + dz;
                buf[m++] = v[j][0];
                buf[m++] = v[j][1];
                buf[m++] = v[j][2];
            }
        } else {
            double dvx, dvy, dvz;
            dvx = pbc[0] * h_rate[0] + pbc[5] * h_rate[5] + pbc[4] * h_rate[4];
            dvy = pbc[1] * h_rate[1] + pbc[3] * h_rate[3];
            dvz = pbc[2] * h_rate[2];
            for( int i = 0; i < n; i++ ) {
                int j = list[i];
                int m = i * sz;
                buf[m++] = x[j][0] + dx;
                buf[m++] = x[j][1] + dy;
                buf[m++] = x[j][2] + dz;
                if( mask[i] & deform_groupbit ) {
                    buf[m++] = v[j][0] + dvx;
                    buf[m++] = v[j][1] + dvy;
                    buf[m++] = v[j][2] + dvz;
                } else {
                    buf[m++] = v[j][0];
                    buf[m++] = v[j][1];
                    buf[m++] = v[j][2];
                }
            }
        }
    }

    return n * sz;
}

void AtomVecDPDAtomic::unpack_comm_vel( int n, int first, double *buf )
{
    int i, m, last;

    m = 0;
    last = first + n;
    for( i = first; i < last; i++ ) {
        x[i][0] = buf[m++];
        x[i][1] = buf[m++];
        x[i][2] = buf[m++];
        v[i][0] = buf[m++];
        v[i][1] = buf[m++];
        v[i][2] = buf[m++];
    }
}
