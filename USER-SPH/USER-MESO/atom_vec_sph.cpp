#include "stdlib.h"
#include "atom_vec_sph.h"
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "modify.h"
#include "fix.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

#define DELTA 10000

/* ---------------------------------------------------------------------- */

AtomVecSPH::AtomVecSPH( LAMMPS *lmp ) : AtomVec( lmp )
{
    molecular = 0;
    mass_type = 1;

    comm_x_only    = 0; // x + v
    comm_f_only    = 1; // no reverse communication actually
    size_forward   = 3; // x
    size_reverse   = 0; // no reverse communication
    size_border    = 6; // x + tag + type + mask
    size_velocity  = 3;
    size_data_atom = 5; // tag type x
    size_data_vel  = 4;
    xcol_data      = 3;

    atom->e_flag   = 0;
    atom->rho_flag = 1;
    atom->cv_flag  = 0;
    atom->vest_flag = 0;
}

/* ----------------------------------------------------------------------
   grow atom arrays
   n = 0 grows arrays by DELTA
   n > 0 allocates arrays to size n
   ------------------------------------------------------------------------- */

void AtomVecSPH::grow( int n )
{
    if( n == 0 )
        nmax += DELTA;
    else
        nmax = n;
    atom->nmax = nmax;
    if( nmax < 0 || nmax > MAXSMALLINT )
        error->one( FLERR, "Per-processor system is too big" );

    tag = memory->grow( atom->tag, nmax, "atom:tag" );
    type = memory->grow( atom->type, nmax, "atom:type" );
    mask = memory->grow( atom->mask, nmax, "atom:mask" );
    image = memory->grow( atom->image, nmax, "atom:image" );
    x = memory->grow( atom->x, nmax, 3, "atom:x" );
    v = memory->grow( atom->v, nmax, 3, "atom:v" );
    f = memory->grow( atom->f, nmax * comm->nthreads, 3, "atom:f" );
    rho = memory->grow( atom->rho, nmax, "atom:rho" );

    if( atom->nextra_grow )
        for( int iextra = 0; iextra < atom->nextra_grow; iextra++ )
            modify->fix[atom->extra_grow[iextra]]->grow_arrays( nmax );
}

/* ----------------------------------------------------------------------
   reset local array ptrs
   ------------------------------------------------------------------------- */

void AtomVecSPH::grow_reset()
{
    tag = atom->tag;
    type = atom->type;
    mask = atom->mask;
    image = atom->image;
    x = atom->x;
    v = atom->v;
    f = atom->f;
    rho = atom->rho;
}

/* ---------------------------------------------------------------------- */

void AtomVecSPH::copy( int i, int j, int delflag )
{
    tag[j] = tag[i];
    type[j] = type[i];
    mask[j] = mask[i];
    image[j] = image[i];
    x[j][0] = x[i][0];
    x[j][1] = x[i][1];
    x[j][2] = x[i][2];
    v[j][0] = v[i][0];
    v[j][1] = v[i][1];
    v[j][2] = v[i][2];

    rho[j] = rho[i];

    if( atom->nextra_grow )
        for( int iextra = 0; iextra < atom->nextra_grow; iextra++ )
            modify->fix[atom->extra_grow[iextra]]->copy_arrays( i, j, delflag );
}

/* ---------------------------------------------------------------------- */

int AtomVecSPH::pack_comm_hybrid( int n, int *list, double *buf )
{
}

/* ---------------------------------------------------------------------- */

int AtomVecSPH::unpack_comm_hybrid( int n, int first, double *buf )
{
}

/* ---------------------------------------------------------------------- */

int AtomVecSPH::pack_border_hybrid( int n, int *list, double *buf )
{
    int i, j, m;

    m = 0;
    for( i = 0; i < n; i++ ) {
        j = list[i];
    }
    return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecSPH::unpack_border_hybrid( int n, int first, double *buf )
{
    int i, m, last;

    m = 0;
    last = first + n;
    for( i = first; i < last; i++ ) {
    }
    return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecSPH::pack_reverse_hybrid( int n, int first, double *buf )
{
    int i, m, last;

    m = 0;
    last = first + n;
    for( i = first; i < last; i++ ) {
    }
    return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecSPH::unpack_reverse_hybrid( int n, int *list, double *buf )
{
    int i, j, m;

    m = 0;
    for( i = 0; i < n; i++ ) {
        j = list[i];
    }
    return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecSPH::pack_comm( int n, int *list, double *buf, int pbc_flag,
                           int *pbc )
{
    int i, j, m;
    double dx, dy, dz;

    m = 0;
    if( pbc_flag == 0 ) {
        for( i = 0; i < n; i++ ) {
            j = list[i];
            buf[m++] = x[j][0];
            buf[m++] = x[j][1];
            buf[m++] = x[j][2];
        }
    } else {
        if( domain->triclinic == 0 ) {
            dx = pbc[0] * domain->xprd;
            dy = pbc[1] * domain->yprd;
            dz = pbc[2] * domain->zprd;
        } else {
            dx = pbc[0] * domain->xprd + pbc[5] * domain->xy + pbc[4] * domain->xz;
            dy = pbc[1] * domain->yprd + pbc[3] * domain->yz;
            dz = pbc[2] * domain->zprd;
        }
        for( i = 0; i < n; i++ ) {
            j = list[i];
            buf[m++] = x[j][0] + dx;
            buf[m++] = x[j][1] + dy;
            buf[m++] = x[j][2] + dz;
        }
    }
    return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecSPH::pack_comm_vel( int n, int *list, double *buf, int pbc_flag,
                               int *pbc )
{
    int i, j, m;
    double dx, dy, dz;

    m = 0;
    if( pbc_flag == 0 ) {
        for( i = 0; i < n; i++ ) {
            j = list[i];
            buf[m++] = x[j][0];
            buf[m++] = x[j][1];
            buf[m++] = x[j][2];
            buf[m++] = v[j][0];
            buf[m++] = v[j][1];
            buf[m++] = v[j][2];
        }
    } else {
        if( domain->triclinic == 0 ) {
            dx = pbc[0] * domain->xprd;
            dy = pbc[1] * domain->yprd;
            dz = pbc[2] * domain->zprd;
        } else {
            dx = pbc[0] * domain->xprd + pbc[5] * domain->xy + pbc[4] * domain->xz;
            dy = pbc[1] * domain->yprd + pbc[3] * domain->yz;
            dz = pbc[2] * domain->zprd;
        }
        for( i = 0; i < n; i++ ) {
            j = list[i];
            buf[m++] = x[j][0] + dx;
            buf[m++] = x[j][1] + dy;
            buf[m++] = x[j][2] + dz;
            buf[m++] = v[j][0];
            buf[m++] = v[j][1];
            buf[m++] = v[j][2];
        }
    }
    return m;
}

/* ---------------------------------------------------------------------- */

void AtomVecSPH::unpack_comm( int n, int first, double *buf )
{
    int i, m, last;

    m = 0;
    last = first + n;
    for( i = first; i < last; i++ ) {
        x[i][0] = buf[m++];
        x[i][1] = buf[m++];
        x[i][2] = buf[m++];
    }
}

/* ---------------------------------------------------------------------- */

void AtomVecSPH::unpack_comm_vel( int n, int first, double *buf )
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

/* ---------------------------------------------------------------------- */

int AtomVecSPH::pack_reverse( int n, int first, double *buf )
{
    int i, m, last;

    m = 0;
    last = first + n;
    for( i = first; i < last; i++ ) {
        buf[m++] = f[i][0];
        buf[m++] = f[i][1];
        buf[m++] = f[i][2];
    }
    return m;
}

/* ---------------------------------------------------------------------- */

void AtomVecSPH::unpack_reverse( int n, int *list, double *buf )
{
    int i, j, m;

    m = 0;
    for( i = 0; i < n; i++ ) {
        j = list[i];
        f[j][0] += buf[m++];
        f[j][1] += buf[m++];
        f[j][2] += buf[m++];
    }
}

/* ---------------------------------------------------------------------- */

int AtomVecSPH::pack_border( int n, int *list, double *buf, int pbc_flag,
                             int *pbc )
{
    int i, j, m;
    double dx, dy, dz;

    m = 0;
    if( pbc_flag == 0 ) {
        for( i = 0; i < n; i++ ) {
            j = list[i];
            buf[m++] = x[j][0];
            buf[m++] = x[j][1];
            buf[m++] = x[j][2];
            buf[m++] = tag[j];
            buf[m++] = type[j];
            buf[m++] = mask[j];
        }
    } else {
        if( domain->triclinic == 0 ) {
            dx = pbc[0] * domain->xprd;
            dy = pbc[1] * domain->yprd;
            dz = pbc[2] * domain->zprd;
        } else {
            dx = pbc[0];
            dy = pbc[1];
            dz = pbc[2];
        }
        for( i = 0; i < n; i++ ) {
            j = list[i];
            buf[m++] = x[j][0] + dx;
            buf[m++] = x[j][1] + dy;
            buf[m++] = x[j][2] + dz;
            buf[m++] = tag[j];
            buf[m++] = type[j];
            buf[m++] = mask[j];
        }
    }

    if( atom->nextra_border )
        for( int iextra = 0; iextra < atom->nextra_border; iextra++ )
            m += modify->fix[atom->extra_border[iextra]]->pack_border( n, list, &buf[m] );

    return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecSPH::pack_border_vel( int n, int *list, double *buf, int pbc_flag,
                                 int *pbc )
{
    int i, j, m;
    double dx, dy, dz;

    m = 0;
    if( pbc_flag == 0 ) {
        for( i = 0; i < n; i++ ) {
            j = list[i];
            buf[m++] = x[j][0];
            buf[m++] = x[j][1];
            buf[m++] = x[j][2];
            buf[m++] = tag[j];
            buf[m++] = type[j];
            buf[m++] = mask[j];
            buf[m++] = v[j][0];
            buf[m++] = v[j][1];
            buf[m++] = v[j][2];
        }
    } else {
        if( domain->triclinic == 0 ) {
            dx = pbc[0] * domain->xprd;
            dy = pbc[1] * domain->yprd;
            dz = pbc[2] * domain->zprd;
        } else {
            dx = pbc[0];
            dy = pbc[1];
            dz = pbc[2];
        }
        for( i = 0; i < n; i++ ) {
            j = list[i];
            buf[m++] = x[j][0] + dx;
            buf[m++] = x[j][1] + dy;
            buf[m++] = x[j][2] + dz;
            buf[m++] = tag[j];
            buf[m++] = type[j];
            buf[m++] = mask[j];
            buf[m++] = v[j][0];
            buf[m++] = v[j][1];
            buf[m++] = v[j][2];
        }
    }

    if( atom->nextra_border )
        for( int iextra = 0; iextra < atom->nextra_border; iextra++ )
            m += modify->fix[atom->extra_border[iextra]]->pack_border( n, list, &buf[m] );

    return m;
}

/* ---------------------------------------------------------------------- */

void AtomVecSPH::unpack_border( int n, int first, double *buf )
{
    int i, m, last;

    m = 0;
    last = first + n;
    for( i = first; i < last; i++ ) {
        if( i == nmax )
            grow( 0 );
        x[i][0] = buf[m++];
        x[i][1] = buf[m++];
        x[i][2] = buf[m++];
        tag[i] = static_cast<int>( buf[m++] );
        type[i] = static_cast<int>( buf[m++] );
        mask[i] = static_cast<int>( buf[m++] );
    }

    if( atom->nextra_border )
        for( int iextra = 0; iextra < atom->nextra_border; iextra++ )
            m += modify->fix[atom->extra_border[iextra]]->
                 unpack_border( n, first, &buf[m] );
}

/* ---------------------------------------------------------------------- */

void AtomVecSPH::unpack_border_vel( int n, int first, double *buf )
{
    int i, m, last;

    m = 0;
    last = first + n;
    for( i = first; i < last; i++ ) {
        if( i == nmax )
            grow( 0 );
        x[i][0] = buf[m++];
        x[i][1] = buf[m++];
        x[i][2] = buf[m++];
        tag[i] = static_cast<int>( buf[m++] );
        type[i] = static_cast<int>( buf[m++] );
        mask[i] = static_cast<int>( buf[m++] );
        v[i][0] = buf[m++];
        v[i][1] = buf[m++];
        v[i][2] = buf[m++];
    }

    if( atom->nextra_border )
        for( int iextra = 0; iextra < atom->nextra_border; iextra++ )
            m += modify->fix[atom->extra_border[iextra]]->
                 unpack_border( n, first, &buf[m] );
}

/* ----------------------------------------------------------------------
   pack data for atom I for sending to another proc
   xyz must be 1st 3 values, so comm::exchange() can test on them
   ------------------------------------------------------------------------- */

int AtomVecSPH::pack_exchange( int i, double *buf )
{
    int m = 1;
    buf[m++] = x[i][0];
    buf[m++] = x[i][1];
    buf[m++] = x[i][2];
    buf[m++] = v[i][0];
    buf[m++] = v[i][1];
    buf[m++] = v[i][2];
    buf[m++] = tag[i];
    buf[m++] = type[i];
    buf[m++] = mask[i];
    buf[m] = 0.0;      // for valgrind
    *( ( tagint * ) &buf[m++] ) = image[i];

    if( atom->nextra_grow )
        for( int iextra = 0; iextra < atom->nextra_grow; iextra++ )
            m += modify->fix[atom->extra_grow[iextra]]->pack_exchange( i, &buf[m] );

    buf[0] = m;
    return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecSPH::unpack_exchange( double *buf )
{
    int nlocal = atom->nlocal;
    if( nlocal == nmax )
        grow( 0 );

    int m = 1;
    x[nlocal][0] = buf[m++];
    x[nlocal][1] = buf[m++];
    x[nlocal][2] = buf[m++];
    v[nlocal][0] = buf[m++];
    v[nlocal][1] = buf[m++];
    v[nlocal][2] = buf[m++];
    tag[nlocal] = static_cast<int>( buf[m++] );
    type[nlocal] = static_cast<int>( buf[m++] );
    mask[nlocal] = static_cast<int>( buf[m++] );
    image[nlocal] = *( ( tagint * ) &buf[m++] );

    if( atom->nextra_grow )
        for( int iextra = 0; iextra < atom->nextra_grow; iextra++ )
            m += modify->fix[atom->extra_grow[iextra]]-> unpack_exchange( nlocal,
                    &buf[m] );

    atom->nlocal++;
    return m;
}

/* ----------------------------------------------------------------------
   size of restart data for all atoms owned by this proc
   include extra data stored by fixes
   ------------------------------------------------------------------------- */

int AtomVecSPH::size_restart()
{
    int i;

    int nlocal = atom->nlocal;
    int n = 17 * nlocal; // 11 + rho + e + cv + vest[3]

    if( atom->nextra_restart )
        for( int iextra = 0; iextra < atom->nextra_restart; iextra++ )
            for( i = 0; i < nlocal; i++ )
                n += modify->fix[atom->extra_restart[iextra]]->size_restart( i );

    return n;
}

/* ----------------------------------------------------------------------
   pack atom I's data for restart file including extra quantities
   xyz must be 1st 3 values, so that read_restart can test on them
   molecular types may be negative, but write as positive
   ------------------------------------------------------------------------- */

int AtomVecSPH::pack_restart( int i, double *buf )
{
    int m = 1;
    buf[m++] = x[i][0];
    buf[m++] = x[i][1];
    buf[m++] = x[i][2];
    buf[m++] = tag[i];
    buf[m++] = type[i];
    buf[m++] = mask[i];
    buf[m] = 0.0;      // for valgrind
    *( ( tagint * ) &buf[m++] ) = image[i];
    buf[m++] = v[i][0];
    buf[m++] = v[i][1];
    buf[m++] = v[i][2];

    if( atom->nextra_restart )
        for( int iextra = 0; iextra < atom->nextra_restart; iextra++ )
            m += modify->fix[atom->extra_restart[iextra]]->pack_restart( i, &buf[m] );

    buf[0] = m;
    return m;
}

/* ----------------------------------------------------------------------
   unpack data for one atom from restart file including extra quantities
   ------------------------------------------------------------------------- */

int AtomVecSPH::unpack_restart( double *buf )
{
    int nlocal = atom->nlocal;
    if( nlocal == nmax ) {
        grow( 0 );
        if( atom->nextra_store )
            memory->grow( atom->extra, nmax, atom->nextra_store, "atom:extra" );
    }

    int m = 1;
    x[nlocal][0] = buf[m++];
    x[nlocal][1] = buf[m++];
    x[nlocal][2] = buf[m++];
    tag[nlocal] = static_cast<int>( buf[m++] );
    type[nlocal] = static_cast<int>( buf[m++] );
    mask[nlocal] = static_cast<int>( buf[m++] );
    image[nlocal] = *( ( tagint * ) &buf[m++] );
    v[nlocal][0] = buf[m++];
    v[nlocal][1] = buf[m++];
    v[nlocal][2] = buf[m++];

    double **extra = atom->extra;
    if( atom->nextra_store ) {
        int size = static_cast<int>( buf[0] ) - m;
        for( int i = 0; i < size; i++ )
            extra[nlocal][i] = buf[m++];
    }

    atom->nlocal++;
    return m;
}

/* ----------------------------------------------------------------------
   create one atom of itype at coord
   set other values to defaults
   ------------------------------------------------------------------------- */

void AtomVecSPH::create_atom( int itype, double *coord )
{
    int nlocal = atom->nlocal;
    if( nlocal == nmax )
        grow( 0 );

    tag[nlocal] = 0;
    type[nlocal] = itype;
    x[nlocal][0] = coord[0];
    x[nlocal][1] = coord[1];
    x[nlocal][2] = coord[2];
    mask[nlocal] = 1;
    image[nlocal] = ( ( tagint ) IMGMAX << IMG2BITS ) |
                    ( ( tagint ) IMGMAX << IMGBITS ) | IMGMAX;
    v[nlocal][0] = 0.0;
    v[nlocal][1] = 0.0;
    v[nlocal][2] = 0.0;
    rho[nlocal] = 0.0;

    atom->nlocal++;
}

/* ----------------------------------------------------------------------
   unpack one line from Atoms section of data file
   initialize other atom quantities
   ------------------------------------------------------------------------- */

void AtomVecSPH::data_atom( double *coord, tagint imagetmp, char **values )
{
    int nlocal = atom->nlocal;
    if( nlocal == nmax )
        grow( 0 );

    tag[nlocal] = atoi( values[0] );
    if( tag[nlocal] <= 0 )
        error->one( FLERR, "Invalid atom ID in Atoms section of data file" );

    type[nlocal] = atoi( values[1] );
    if( type[nlocal] <= 0 || type[nlocal] > atom->ntypes )
        error->one( FLERR, "Invalid atom type in Atoms section of data file" );

    rho[nlocal] = atof( values[2] );

    x[nlocal][0] = coord[0];
    x[nlocal][1] = coord[1];
    x[nlocal][2] = coord[2];

    image[nlocal] = imagetmp;

    mask[nlocal] = 1;
    v[nlocal][0] = 0.0;
    v[nlocal][1] = 0.0;
    v[nlocal][2] = 0.0;

    atom->nlocal++;
}

/* ----------------------------------------------------------------------
   unpack hybrid quantities from one line in Atoms section of data file
   initialize other atom quantities for this sub-style
   ------------------------------------------------------------------------- */

int AtomVecSPH::data_atom_hybrid( int nlocal, char **values )
{

    error->all( FLERR, "<MESO> no hybrid atom using AtomVecSPH" );
    rho[nlocal] = atof( values[0] );

    return 3;
}

/* ----------------------------------------------------------------------
   pack atom info for data file including 3 image flags
------------------------------------------------------------------------- */

void AtomVecSPH::pack_data( double **buf )
{
    int nlocal = atom->nlocal;
    for( int i = 0; i < nlocal; i++ ) {
        buf[i][0] = tag[i];
        buf[i][1] = type[i];
        buf[i][2] = x[i][0];
        buf[i][3] = x[i][1];
        buf[i][4] = x[i][2];
        buf[i][5] = ( image[i] & IMGMASK ) - IMGMAX;
        buf[i][6] = ( image[i] >> IMGBITS & IMGMASK ) - IMGMAX;
        buf[i][7] = ( image[i] >> IMG2BITS ) - IMGMAX;
    }
}

/* ----------------------------------------------------------------------
   pack hybrid atom info for data file
------------------------------------------------------------------------- */

int AtomVecSPH::pack_data_hybrid( int i, double *buf )
{
    error->all( FLERR, "<MESO> no hybrid atom using AtomVecSPH" );
    return 0;
}

/* ----------------------------------------------------------------------
   write atom info to data file including 3 image flags
------------------------------------------------------------------------- */

void AtomVecSPH::write_data( FILE *fp, int n, double **buf )
{
    for( int i = 0; i < n; i++ )
        fprintf( fp, "%d %d %-1.16e %-1.16e %-1.16e %d %d %d\n",
                 ( int ) buf[i][0], ( int ) buf[i][1],
                 buf[i][2], buf[i][3], buf[i][4],
                 ( int ) buf[i][5], ( int ) buf[i][6], ( int ) buf[i][7] );
}

/* ----------------------------------------------------------------------
   write hybrid atom info to data file
------------------------------------------------------------------------- */

int AtomVecSPH::write_data_hybrid( FILE *fp, double *buf )
{
    error->all( FLERR, "<MESO> no hybrid atom using AtomVecSPH" );
    return 0;
}

/* ----------------------------------------------------------------------
   return # of bytes of allocated memory
   ------------------------------------------------------------------------- */

bigint AtomVecSPH::memory_usage()
{
    bigint bytes = 0;

    if( atom->memcheck( "tag" ) )
        bytes += memory->usage( tag, nmax );
    if( atom->memcheck( "type" ) )
        bytes += memory->usage( type, nmax );
    if( atom->memcheck( "mask" ) )
        bytes += memory->usage( mask, nmax );
    if( atom->memcheck( "image" ) )
        bytes += memory->usage( image, nmax );
    if( atom->memcheck( "x" ) )
        bytes += memory->usage( x, nmax, 3 );
    if( atom->memcheck( "v" ) )
        bytes += memory->usage( v, nmax, 3 );
    if( atom->memcheck( "f" ) )
        bytes += memory->usage( f, nmax * comm->nthreads, 3 );
    if( atom->memcheck( "rho" ) )
        bytes += memory->usage( rho, nmax );

    return bytes;
}
