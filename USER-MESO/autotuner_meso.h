#ifndef LMP_MESO_AUTOTUNER
#define LMP_MESO_AUTOTUNER

#include <algorithm>
#include "util_meso.h"
#include "math_meso.h"

namespace LAMMPS_NS
{

namespace ATPARAM
{
const static double memory = 0.618;
const static int diff = 10;
}

struct ThreadStat {
    ThreadStat() : p_buffer( 0 ), value( 0 ) {}
    inline double val() const
    {
        return value;
    }
    inline bool push( double x )
    {
        buffer[p_buffer++] = x;
        if( p_buffer >= ATPARAM::diff ) {
            // find minimum value
            double mini = std::numeric_limits<double>::max();
            for( int i = 0 ; i < ATPARAM::diff ; i++ ) mini = std::min( mini, buffer[i] );
            // corrected mean
            int n = 0;
            double sum = 0;
            for( int i = 0 ; i < ATPARAM::diff ; i++ ) {
                if( buffer[i] < mini * 1.618 ) {
                    sum += buffer[i];
                    n++;
                }
            }
            double value_next = sum / n;
            if( value > 0 ) value = value * ATPARAM::memory + value_next * ( 1.0 - ATPARAM::memory );
            else value = value_next;
            p_buffer = 0;
            return true;
        } else return false;
    }
protected:
    int p_buffer;
    double value, buffer[ATPARAM::diff];
};

class ThreadTuner
{
public:
    ThreadTuner() : lower( 1 ), upper( omp_get_max_threads() ), i( lower ), level( 0 ), experience( upper + 1, false ), grid( upper + 1 )
    {
        fprintf( stderr, "<MESO> Default constructor called?\n" );
    }
    ThreadTuner( size_t lower_, std::size_t upper_, std::string tag_ ) : lower( lower_ ), upper( upper_ ), i( lower_ ), level( 0 ), experience( upper + 1, false ), grid( upper_ + 1 ), tag( tag_ )
    {
    }

    inline void learn( std::size_t n, double t )
    {
        if( grid[n].push( t ) ) {
            // check for experience increase
            experience[n] = true;
            bool levelup = true;
            for( int p = lower ; p <= upper ; p++ ) levelup = levelup && experience[p];
            if( levelup ) {
                level++;
                experience.assign( upper + 1, false );
            }
            // re-determine the optimal configuration
            double fastest = std::numeric_limits<double>::max();
            for( int p = lower ; p <= upper ; p++ ) {
                if( grid[p].val() < fastest ) {
                    i = p;
                    fastest = grid[p].val();
                }
            }
        }
    }
    inline std::size_t bet() const
    {
        return i;
    }
    inline std::size_t lv() const
    {
        return level;
    }
protected:
    std::size_t lower, upper, i, level;
    std::string tag;
    std::vector<bool> experience;
    std::vector<ThreadStat> grid;
};

}

#endif
