/*
 * nx.cpp
 *
 *  Created on: Jan 5, 2012
 *      Author: Eugen Wintersberger
 */

extern "C"{
#include<Python.h>
#include<numpy/arrayobject.h>
}

#include <boost/python.hpp>
#include <iostream>
#include <sstream>

using namespace boost::python;


//! \endcond

extern void wrap_factory();
extern void wrap_group();
extern void wrap_dataset();
extern void wrap_dataitem();
extern void exception_registration();
extern void wrap_attribute();


//=================implementation of the python extension======================
BOOST_PYTHON_MODULE(cdmacore)
{
    
    //this is absolutely necessary - otherwise the nympy API functions do not
    //work.
    import_array();

    //call wrappers
    wrap_factory();
    wrap_dataset();
    wrap_group();
    wrap_dataitem();
    wrap_attribute();
    exception_registration();

}
