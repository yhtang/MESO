/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "mpi.h"
#include "lammps.h"
#include "input.h"
#include "string.h"
#include <string>
#include <fstream>
#include <vector>

using namespace LAMMPS_NS;

/* ----------------------------------------------------------------------
   main program to drive LAMMPS
------------------------------------------------------------------------- */

void main_default( int argc, char ** argv );
void main_ensemble( int argc, char ** argv );

int main( int argc, char ** argv ) {
    bool ensemble_run = false;
    for ( int i = 0; i < argc; i++ ) if ( !strcmp( argv[i], "-ensemble" ) ) ensemble_run = true;
    MPI_Init( &argc, &argv );

    if ( ensemble_run ) main_ensemble( argc, argv );
    else main_default( argc, argv );

    MPI_Finalize();
    return 0;
}

void main_default( int argc, char ** argv ) {
    LAMMPS * lammps = new LAMMPS( argc, argv, MPI_COMM_WORLD );
    lammps->input->file();
    delete lammps;
}

void main_ensemble( int argc, char ** argv ) {
    int case_size = 0;
    std::string arg_file;
    for ( int i = 0; i < argc; i++ ) {
        if ( !strcmp( argv[i], "-arg_file" ) ) {
            arg_file = argv[++i];
            continue;
        }
        if ( !strcmp( argv[i], "-case_size" ) ) {
            case_size = atoi( argv[++i] );
            continue;
        }
    }

    if ( case_size == 0 ) {
        printf( "<MESO> -case_size for ensemble run not specified.\n" );
        return;
    }
    std::ifstream farg( arg_file.c_str() );
    if ( farg.eof() || farg.fail() ) {
        printf( "<MESO> argument file %s cannot be opened.\n", arg_file.c_str() );
    }

    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    int case_id = rank / case_size;
    MPI_Comm domain;
    MPI_Comm_split( MPI_COMM_WORLD, case_id, rank, &domain );

    int cmdline_maxlen = 1024 * 1024;
    char * cmdline = new char[cmdline_maxlen];
    // skip the arguments for cases in front of me
    for ( int i = 0; i < case_id; i++ ) {
        farg.getline( cmdline, cmdline_maxlen );
    }
    farg.getline( cmdline, cmdline_maxlen );
    int cmdline_len = strlen( cmdline );
    std::vector<char *> cmd_argv;
    // LAMMPS ignores the 0th argument
    char * ignore = new char [10];
    memset(ignore, 0, 10 );
    cmd_argv.push_back( ignore );
    // split the command line string
    std::vector<char> buffer;
    for ( int i = 0; i < cmdline_len; i++ ) {
        if ( !isspace( cmdline[i] ) ) buffer.push_back( cmdline[i] );
        if ( isspace( cmdline[i] ) || i == cmdline_len -1 ) {
            if ( buffer.size() ) {
                char * arg = new char[buffer.size() + 1 ];
                memset( arg, 0, buffer.size() + 1 );
                for ( int j = 0; j < buffer.size(); j++ ) arg[j] = buffer[j];
                cmd_argv.push_back( arg );
                buffer.clear();
            }
        }
    }

    int len;
    char node_name[1024];
    MPI_Get_processor_name( node_name, &len );
    printf( "rank %d nid %s processing command line '%s' [%d arguments]\n", rank, node_name, cmdline, cmd_argv.size() );

    LAMMPS * lammps = new LAMMPS( cmd_argv.size(), cmd_argv.data(), domain );
    lammps->input->file();

    for ( int i = 0; i < cmd_argv.size(); i++ ) delete [] cmd_argv[i];

    delete lammps;
}
