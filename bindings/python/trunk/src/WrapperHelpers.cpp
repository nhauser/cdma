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
    for(size_t i=0;i<array.rank();i++) dims[i] = array.shape()[i];

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

