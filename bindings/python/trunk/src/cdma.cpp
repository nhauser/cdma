/*
 * (c) Copyright 2012 DESY, Eugen Wintersberger <eugen.wintersberger@desy.de>
 *
 * This file is part of cdma-python.
 *
 * cdma-python is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 *
 * cdma-python is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with cmda-python.  If not, see <http://www.gnu.org/licenses/>.
 *************************************************************************
 *
 * Created on: Jun 26, 2011
 *     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
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
extern void wrap_dimension();


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
    wrap_dimension();
    exception_registration();

}
