#include "lmptype.h"
#include "mpi.h"
#include "math.h"
#include "string.h"
#include "stdio.h"
#include "stdlib.h"
#include "universe.h"
#include "atom.h"
#include "atom_vec.h"
#include "force.h"
#include "pair.h"
#include "domain.h"
#include "neighbor.h"
#include "group.h"
#include "modify.h"
#include "fix.h"
#include "compute.h"
#include "output.h"
#include "dump.h"
#include "procmap.h"
#include "math_extra.h"
#include "error.h"
#include "update.h"
#include "memory.h"

#include "atom_meso.h"
#include "comm_meso.h"

using namespace LAMMPS_NS;

#define BUFFACTOR 1.5
#define BUFMIN 1000
#define BUFEXTRA 1000
#define BIG 1.0e20

enum {SINGLE, MULTI};
enum {MULTIPLE};                  // same as in ProcMap
enum {ONELEVEL, TWOLEVEL, NUMA, CUSTOM};
enum {CART, CARTREORDER, XYZ};

void MesoComm::borders() {
    int n, iswap, dim, ineed, twoneed, smax, rmax;
    int nsend, nrecv, sendflag, nfirst, nlast, ngroup;
    double lo, hi;
    int * type;
    double ** x;
    double * buf, *mlo, *mhi;
    MPI_Request request;
    MPI_Status status;
    AtomVec * avec = atom->avec;

    // do swaps over all 3 dimensions

    iswap = 0;
    smax = rmax = 0;

    std::set<int> borders;
    static ThreadTuner & o = meso_device->tuner( "MesoComm::pick_border" );
    double t1, t2;
    if ( meso_device->warmed_up() ) t1 = meso_device->get_time_omp();

    for ( dim = 0; dim < 3; dim++ ) {
        nlast = meso_atom->n_bulk;
        twoneed = 2 * maxneed[dim];
        for ( ineed = 0; ineed < twoneed; ineed++ ) {

            // find atoms within slab boundaries lo/hi using <= and >=
            // check atoms between nfirst and nlast
            //     for first swaps in a dim, check owned and ghost
            //     for later swaps in a dim, only check newly arrived ghosts
            // store sent atom indices in list for use in future timesteps

            x = atom->x;
            if ( style == SINGLE ) {
                lo = slablo[iswap];
                hi = slabhi[iswap];
            } else {
                type = atom->type;
                mlo = multilo[iswap];
                mhi = multihi[iswap];
            }
            if ( ineed % 2 == 0 ) {
                nfirst = nlast;
                nlast = atom->nlocal + atom->nghost;
            }

            nsend = 0;

            // sendflag = 0 if I do not send on this swap
            // sendneed test indicates receiver no longer requires data
            // e.g. due to non-PBC or non-uniform sub-domains

            if ( ineed / 2 >= sendneed[dim][ineed % 2] ) sendflag = 0;
            else sendflag = 1;

            // find send atoms according to SINGLE vs MULTI
            // all atoms eligible versus atoms in bordergroup
            // only need to limit loop to bordergroup for first sends (ineed < 2)
            // on these sends, break loop in two: owned (in group) and ghost

            if ( sendflag ) {
                if ( !bordergroup || ineed >= 2 ) {
                    if ( style == SINGLE ) {
                        pick_border( nfirst, nlast, iswap, nsend, pred_border_single( dim, lo, hi, x ), o.bet() );
                    } else {
                        pick_border( nfirst, nlast, iswap, nsend, pred_border_multi( dim, type, mlo, mhi, x ), o.bet() );
                    }
                } else {
                    if ( style == SINGLE ) {
                        ngroup = atom->nfirst;
                        pick_border( 0, ngroup, iswap, nsend, pred_border_single( dim, lo, hi, x ), o.bet() );
                        pick_border( atom->nlocal, nlast, iswap, nsend, pred_border_single( dim, lo, hi, x ), o.bet() );
                    } else {
                        ngroup = atom->nfirst;
                        pick_border( 0, ngroup, iswap, nsend, pred_border_multi( dim, type, mlo, mhi, x ), o.bet() );
                        pick_border( atom->nlocal, nlast, iswap, nsend, pred_border_multi( dim, type, mlo, mhi, x ), o.bet() );
                    }
                }
            }

            // pack up list of border atoms
            if ( nsend * size_border > maxsend ) grow_send( nsend * size_border, 0 );
            if ( ghost_velocity )
                n = avec->pack_border_vel( nsend, sendlist[iswap], buf_send,
                                           pbc_flag[iswap], pbc[iswap] );
            else
                n = avec->pack_border( nsend, sendlist[iswap], buf_send,
                                       pbc_flag[iswap], pbc[iswap] );

            // swap atoms with other proc
            // no MPI calls except SendRecv if nsend/nrecv = 0
            // put incoming ghosts at end of my atom arrays
            // if swapping with self, simply copy, no messages

            if ( sendproc[iswap] != me ) {
                MPI_Sendrecv( &nsend, 1, MPI_INT, sendproc[iswap], 0,
                              &nrecv, 1, MPI_INT, recvproc[iswap], 0, world, &status );
                if ( nrecv * size_border > maxrecv ) grow_recv( nrecv * size_border );
                if ( nrecv ) MPI_Irecv( buf_recv, nrecv * size_border, MPI_DOUBLE,
                                            recvproc[iswap], 0, world, &request );
                if ( n ) MPI_Send( buf_send, n, MPI_DOUBLE, sendproc[iswap], 0, world );
                if ( nrecv ) MPI_Wait( &request, &status );
                buf = buf_recv;
            } else {
                nrecv = nsend;
                buf = buf_send;
            }

            // unpack buffer

            if ( ghost_velocity )
                avec->unpack_border_vel( nrecv, atom->nlocal + atom->nghost, buf );
            else
                avec->unpack_border( nrecv, atom->nlocal + atom->nghost, buf );

            // set all pointers & counters

            smax = MAX( smax, nsend );
            rmax = MAX( rmax, nrecv );
            sendnum[iswap] = nsend;
            recvnum[iswap] = nrecv;
            size_forward_recv[iswap] = nrecv * size_forward;
            size_reverse_send[iswap] = nrecv * size_reverse;
            size_reverse_recv[iswap] = nsend * size_reverse;
            firstrecv[iswap] = atom->nlocal + atom->nghost;
            atom->nghost += nrecv;
            iswap++;
        }
    }

    if ( meso_device->warmed_up() ) {
        t2 = meso_device->get_time_omp();
        o.learn( o.bet(), t2 - t1 );
    }

    // insure send/recv buffers are long enough for all forward & reverse comm

    int max = MAX( maxforward * smax, maxreverse * rmax );
    if ( max > maxsend ) grow_send( max, 0 );
    max = MAX( maxforward * rmax, maxreverse * smax );
    if ( max > maxrecv ) grow_recv( max );

    // not necessary as in contrast to the CPU implementation as it is done explicitly
    // in MVV::run()
    if ( map_style ) atom->map_set();
}

void MesoComm::borderness( std::vector<int> & is_border_ ) {
    // find atoms within slab boundaries lo/hi using <= and >=
    // check atoms between nfirst and nlast
    //     for first swaps in a dim, check owned and ghost
    //     for later swaps in a dim, only check newly arrived ghosts
    // store sent atom indices in list for use in future timesteps

    int   *  type = atom->type;
    double ** x    = atom->x;

    std::vector<int> is_border( atom->nlocal, 0 );
    //is_border.assign( atom->nlocal, 0 );

    static ThreadTuner & o = meso_device->tuner( "MesoComm::borderness" );
    size_t ntd = o.bet();
    double t1, t2;
    if ( meso_device->warmed_up() ) t1 = meso_device->get_time_omp();

    #pragma omp parallel
    {
        for ( int dim = 0, iswap = 0; dim < 3; dim++ ) {
            for ( int ineed = 0; ineed < 2 * maxneed[dim]; ineed++ ) {
                double lo, hi;
                double * mlo, *mhi;

                if ( style == SINGLE ) {
                    lo = slablo[iswap];
                    hi = slabhi[iswap];
                } else {
                    mlo = multilo[iswap];
                    mhi = multihi[iswap];
                }

                // sendflag = 0 if I do not send on this swap
                // sendneed test indicates receiver no longer requires data
                // e.g. due to non-PBC or non-uniform sub-domains

                int sendflag = ( ineed / 2 < sendneed[dim][ineed % 2] ) ? 1 : 0;

                if ( sendflag ) {
                    if ( !bordergroup || ineed >= 2 ) {
                        if ( style == SINGLE ) {
                            get_borderness( is_border, 0, atom->nlocal, pred_border_single( dim, lo, hi, x ), ntd );
                        } else {
                            get_borderness( is_border, 0, atom->nlocal, pred_border_multi( dim, type, mlo, mhi, x ), ntd );
                        }
                    } else {
                        if ( style == SINGLE ) {
                            get_borderness( is_border, 0, atom->nfirst, pred_border_single( dim, lo, hi, x ), ntd );
                        } else {
                            get_borderness( is_border, 0, atom->nfirst, pred_border_multi( dim, type, mlo, mhi, x ), ntd );
                        }
                    }
                }

                iswap++;
            }
        }
    }

    if ( meso_device->warmed_up() ) {
        t2 = meso_device->get_time_omp();
        o.learn( ntd, t2 - t1 );
    }

    is_border_ = is_border;
}

void MesoComm::exchange() {
    int m, nlocal;
    double lo, hi, value;
    double ** x;
    double * sublo, *subhi, *buf;
    MPI_Request request[2];
    MPI_Status status[2];
    AtomVec * avec = atom->avec;

    // clear global->local map for owned and ghost atoms
    // b/c atoms migrate to new procs in exchange() and
    //     new ghosts are created in borders()
    // map_set() is done at end of borders()
    // clear ghost count and any ghost bonus data internal to AtomVec

    if ( map_style ) atom->map_clear();
    atom->nghost = 0;
    atom->avec->clear_bonus();

    // insure send buf is large enough for single atom
    // fixes can change per-atom size requirement on-the-fly

    int bufextra_old = bufextra;
    maxexchange = maxexchange_atom + maxexchange_fix;
    bufextra = maxexchange + BUFEXTRA;
    if ( bufextra > bufextra_old )
        memory->grow( buf_send, maxsend + bufextra, "comm:buf_send" );
    // subbox bounds for orthogonal or triclinic

    if ( triclinic == 0 ) {
        sublo = domain->sublo;
        subhi = domain->subhi;
    } else {
        sublo = domain->sublo_lamda;
        subhi = domain->subhi_lamda;
    }

    // loop over dimensions

    for ( int dim = 0; dim < 3; dim++ ) {

        // fill buffer with atoms leaving my box, using < and >=
        // when atom is deleted, fill it in with last atom

        #if 1

        x = atom->x;
        lo = sublo[dim];
        hi = subhi[dim];
        nlocal = atom->nlocal;
        int i = 0;
        static double * buf_l = NULL, *buf_r = NULL;
        static int buf_sz_send_l = 0, buf_sz_send_r = 0;
        int nsend_l = 0, nsend_r = 0;
        int msg_sz_send_l = 0,
            msg_sz_send_r = 0;
        int msg_sz_recv = 0,
            msg_sz_recv_l,
            msg_sz_recv_r;

        if ( !buf_l ) memory->grow( buf_l, bufextra, "comm:buf_send_l" );
        if ( !buf_r ) memory->grow( buf_r, bufextra, "comm:buf_send_r" );

        double mid = 0.5 * ( lo + hi );
        while ( i < nlocal ) {
            if ( x[i][dim] >= hi || x[i][dim] < lo ) {
                double dist[3];
                dist[dim] = x[i][dim] - mid;
                domain->minimum_image( dist[0], dist[1], dist[2] );
                if ( dist[dim] < 0 ) {
                    if ( msg_sz_send_l > buf_sz_send_l ) {
                        buf_sz_send_l = BUFFACTOR * msg_sz_send_l;
                        memory->grow( buf_l, buf_sz_send_l + bufextra, "comm:buf_send_l" );
                    }
                    msg_sz_send_l += avec->pack_exchange( i, buf_l + msg_sz_send_l );
                    //printf("rank %d packing particle at (%lf,%lf,%lf) for neighbor rank %d\n", me, x[i][0], x[i][1], x[i][2], procneigh[dim][0] );
                    avec->copy( nlocal - 1, i, 1 );
                    ++nsend_l;
                    nlocal--;
                }  else {
                    if ( msg_sz_send_r > buf_sz_send_r ) {
                        buf_sz_send_r = BUFFACTOR * msg_sz_send_r;
                        memory->grow( buf_r, buf_sz_send_r + bufextra, "comm:buf_send_r" );
                    }
                    msg_sz_send_r += avec->pack_exchange( i, buf_r + msg_sz_send_r );
                    //printf("rank %d packing particle at (%lf,%lf,%lf) for neighbor rank %d\n", me, x[i][0], x[i][1], x[i][2], procneigh[dim][1] );
                    avec->copy( nlocal - 1, i, 1 );
                    ++nsend_r;
                    nlocal--;
                }
            } else {
                i++;
            }
        }

        #if 0
        {
            MPI_Barrier( MPI_COMM_WORLD );
            char info[512];
            sprintf( info, "rank %d [%lf-%lf DIM %d] packed %d particles, %d left, %d right\n", me, lo, hi, dim, atom->nlocal - nlocal, nsend_l, nsend_r );
            if ( me == 0 ) {
                MPI_Status stat;
                fprintf( stderr, info );
                for ( int i = 1 ; i < nprocs ; i++ ) {
                    MPI_Recv( info, 512, MPI_CHAR, i, i, MPI_COMM_WORLD, &stat );
                    fprintf( stderr, info );
                }
            } else {
                MPI_Send( info, 512, MPI_CHAR, 0, me, MPI_COMM_WORLD );
            }
            MPI_Barrier( MPI_COMM_WORLD );
        }
        #endif

        atom->nlocal = nlocal;

        if ( msg_sz_send_l + msg_sz_send_r > maxsend ) grow_send( msg_sz_send_l + msg_sz_send_r, 1 );
        memcpy( buf_send           , buf_l, msg_sz_send_l * sizeof( double ) );
        memcpy( buf_send + msg_sz_send_l, buf_r, msg_sz_send_r * sizeof( double ) );

        // send/recv atoms in both directions
        // if 1 proc in dimension, no send/recv, set recv buf to send buf
        // if 2 procs in dimension, single send/recv
        // if more than 2 procs in dimension, send/recv to both neighbors

        if ( procgrid[dim] == 1 ) {
            msg_sz_recv = msg_sz_send_l + msg_sz_send_r;
            buf   = buf_send;
        } else {
            MPI_Sendrecv( &msg_sz_send_l, 1, MPI_INT, procneigh[dim][0], 0,
                          &msg_sz_recv_r, 1, MPI_INT, procneigh[dim][1], 0, world, &status[0] );
            MPI_Sendrecv( &msg_sz_send_r, 1, MPI_INT, procneigh[dim][1], 0,
                          &msg_sz_recv_l, 1, MPI_INT, procneigh[dim][0], 0, world, &status[1] );
            if ( msg_sz_recv_l + msg_sz_recv_r > maxrecv ) grow_recv( msg_sz_recv_l + msg_sz_recv_r );
            msg_sz_recv = msg_sz_recv_l + msg_sz_recv_r;

            if ( procgrid[dim] == 2 ) { // left and right neighbor being the same rank
                MPI_Irecv( buf_recv, msg_sz_recv_l + msg_sz_recv_r, MPI_DOUBLE, procneigh[dim][1], 0, world, &request[0] );
                MPI_Send ( buf_send, msg_sz_send_l + msg_sz_send_r, MPI_DOUBLE, procneigh[dim][0], 0, world );
                MPI_Wait( request, status );
            } else { // left and right neighbors being different ranks
                MPI_Irecv( buf_recv,                 msg_sz_recv_r, MPI_DOUBLE, procneigh[dim][1], 0, world, &request[0] );
                MPI_Irecv( buf_recv + msg_sz_recv_r, msg_sz_recv_l, MPI_DOUBLE, procneigh[dim][0], 0, world, &request[1] );
                MPI_Send ( buf_send,                 msg_sz_send_l, MPI_DOUBLE, procneigh[dim][0], 0, world );
                MPI_Send ( buf_send + msg_sz_send_l, msg_sz_send_r, MPI_DOUBLE, procneigh[dim][1], 0, world );
                MPI_Waitall( 2, request, status );
            }

            buf = buf_recv;
        }

        // check incoming atoms to see if they are in my box
        // if so, add to my list
        m = 0;
        int k = 0;
        int unpacked = 0;
        while ( m < msg_sz_recv ) {
            value = buf[m + dim + 1];
            if ( value >= lo && value < hi ) {
                m += avec->unpack_exchange( &buf[m] );
                ++unpacked;
            } else {
                printf( "rank %d incoming particle at (%lf,%lf,%lf) from rank %d rejected, dim = %d, lo = %lf, hi = %lf\n", me, buf[m + 1], buf[m + 2], buf[m + 3], k < msg_sz_recv_r ? procneigh[dim][1] : procneigh[dim][0], dim, lo, hi );
                m += static_cast<int>( buf[m] );
            }
            ++k;
        }

        #if 0
        {
            MPI_Barrier( MPI_COMM_WORLD );
            char info[512];
            sprintf( info, "rank %d unpacked %d particles\n", me, unpacked );
            if ( me == 0 ) {
                MPI_Status stat;
                fprintf( stderr, info );
                for ( int i = 1 ; i < nprocs ; i++ ) {
                    MPI_Recv( info, 512, MPI_CHAR, i, i, MPI_COMM_WORLD, &stat );
                    fprintf( stderr, info );
                }
            } else {
                MPI_Send( info, 512, MPI_CHAR, 0, me, MPI_COMM_WORLD );
            }
            MPI_Barrier( MPI_COMM_WORLD );
        }
        #endif

        #else
        x = atom->x;
        lo = sublo[dim];
        hi = subhi[dim];
        nlocal = atom->nlocal;

        static ThreadTuner & o = meso_device->tuner( "MesoComm::exchange" );
        size_t ntd = o.bet();
        double t1 = meso_device->get_time_omp();

        static std::set<int> stray;
        static std::vector<std::vector<int> > stray_local;
        if ( stray_local.size() < ntd ) stray_local.resize( ntd );
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            if ( tid < ntd ) {
                std::vector<int> & my_stray = stray_local[ tid ];
                my_stray.clear();
                int2 work = split_work( 0, nlocal, tid, ntd );
                for ( int i = work.x ; i < work.y; i++ ) {
                    if ( x[i][dim] < lo || x[i][dim] >= hi ) my_stray.push_back( i );
                }
            }
        }
        stray.clear();
        for ( int i = 0; i < ntd; i++ ) {
            std::vector<int> & s = stray_local[i];
            for ( int j = 0; j < s.size(); j++ ) stray.insert( s[j] );
        }

        double t2 = meso_device->get_time_omp();
        if ( meso_device->warmed_up() ) o.learn( ntd, t2 - t1 );

        int nsend = 0;
        int p_last = atom->nlocal;
        do {
            --p_last;
        } while ( stray.find( p_last ) != stray.end() );
        for ( std::set<int>::iterator i = stray.begin(); i != stray.end(); i++ ) {
            if ( nsend > maxsend ) grow_send( nsend, 1 );
            nsend += avec->pack_exchange( *i, &buf_send[nsend] );
            if ( *i < p_last ) {
                avec->copy( p_last, *i, 1 );
                do {
                    --p_last;
                } while ( stray.find( p_last ) != stray.end() );
            }
        }

        atom->nlocal = nlocal - stray.size();

        // send/recv atoms in both directions
        // if 1 proc in dimension, no send/recv, set recv buf to send buf
        // if 2 procs in dimension, single send/recv
        // if more than 2 procs in dimension, send/recv to both neighbors

        if ( procgrid[dim] == 1 ) {
            nrecv = nsend;
            buf = buf_send;

        } else {
            MPI_Sendrecv( &nsend, 1, MPI_INT, procneigh[dim][0], 0,
                          &nrecv1, 1, MPI_INT, procneigh[dim][1], 0, world, &status );
            nrecv = nrecv1;
            if ( procgrid[dim] > 2 ) {
                MPI_Sendrecv( &nsend, 1, MPI_INT, procneigh[dim][1], 0,
                              &nrecv2, 1, MPI_INT, procneigh[dim][0], 0, world, &status );
                nrecv += nrecv2;
            }
            if ( nrecv > maxrecv ) grow_recv( nrecv );

            MPI_Irecv( buf_recv, nrecv1, MPI_DOUBLE, procneigh[dim][1], 0,
                       world, &request );
            MPI_Send( buf_send, nsend, MPI_DOUBLE, procneigh[dim][0], 0, world );
            MPI_Wait( &request, &status );

            if ( procgrid[dim] > 2 ) {
                MPI_Irecv( &buf_recv[nrecv1], nrecv2, MPI_DOUBLE, procneigh[dim][0], 0,
                           world, &request );
                MPI_Send( buf_send, nsend, MPI_DOUBLE, procneigh[dim][1], 0, world );
                MPI_Wait( &request, &status );
            }

            buf = buf_recv;
        }

        // check incoming atoms to see if they are in my box
        // if so, add to my list

        m = 0;
        int k = 0;
        while ( m < nrecv ) {
            value = buf[m + dim + 1];
            if ( value >= lo && value < hi ) m += avec->unpack_exchange( &buf[m] );
            else {
                printf( "rank %d incoming particle @%lf,%lf,%lf from rank %d rejected, lo = %lf, hi = %lf\n", me, buf[m + 1], buf[m + 2], buf[m + 3], k < nrecv1 ? procneigh[dim][1] : procneigh[dim][0], lo, hi );
                m += static_cast<int>( buf[m] );
            }
            ++k;
        }

        #endif

    }

    if ( atom->firstgroupname ) atom->first_reorder();
}


