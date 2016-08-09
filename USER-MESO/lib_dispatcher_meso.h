/*
 * lib_dispatcher_meso.h
 *
 *  Created on: Mar 25, 2014
 *      Author: ytang
 */

#ifndef LMP_LIB_DISPATCHER_MESO_H_
#define LMP_LIB_DISPATCHER_MESO_H_

#include <map>
#include <csignal>

template<typename UUID, class FPTR>
struct Dispatcher {
    FPTR dispatch( const UUID id )
    {
        assoc_iter i = table.find( id );
        if( i == table.end() ) raise( SIGSEGV );
        return i->second;
    }
    FPTR operator []( const UUID id )
    {
        return dispatch( id );
    }
    bool link( const UUID id, FPTR parser )
    {
        return table.insert( std::make_pair( id, parser ) ).second;
    }
    bool unlink( const UUID id )
    {
        return table.erase( id ) == 1;
    }
protected:
    typedef typename std::map<UUID, FPTR>::iterator assoc_iter;
    std::map<UUID, FPTR> table;
};

#endif
