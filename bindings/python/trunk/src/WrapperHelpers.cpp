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
 * Created on: Jun 27, 2011
 *     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
 */
#include "WrapperHelpers.hpp"


//-----------------------------------------------------------------------------
void init_numpy()
{
    import_array();
}



//-----------------------------------------------------------------------------
object cdma2numpy_array(const ArrayWrapper &array,bool copyflag)
{
    init_numpy();
    
    //set the dimension of the new array
    npy_intp *dims = new npy_intp[array.rank()]; 
    for(size_t i=0;i<array.rank();i++) 
        dims[i] = array.shape()[i];//[array.rank()-i-1];

    //create the new numpy array
    PyObject *nparray = nullptr;
    if(copyflag)
    {
        nparray = PyArray_SimpleNewFromData(array.rank(),
                                  dims,
                                  typeid2numpytc[array.type()],
                                  const_cast<void *>(array.ptr()));
    }
    else
    {
        nparray = PyArray_SimpleNew(array.rank(),dims,typeid2numpytc[array.type()]);
    }


    if(dims) delete [] dims;
    if(!nparray)
    {
        //THROW EXCEPTION HERE
        std::cerr<<"Error creating new numpy array!"<<std::endl;
    }

    handle<> h(nparray);

    return object(h);
}

