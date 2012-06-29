#include "WrapperHelpers.hpp"

void throw_PyTypeError(const std::string &message)
{
    PyErr_SetString(PyExc_TypeError,message.c_str());
    throw error_already_set();
}

//-----------------------------------------------------------------------------
void init_numpy()
{
    import_array();
}



//-----------------------------------------------------------------------------
object cdma2numpy_array(const ArrayPtr aptr,bool copyflag)
{
    init_numpy();
    //get the type ID of the array
    TypeID tid = typename2typeid[aptr->getValueType().name()];
    //set the dimension of the new array
    npy_intp *dims = new npy_intp[aptr->getRank()]; 
    for(size_t i=0;i<aptr->getRank();i++) 
        dims[i] = aptr->getShape()[aptr->getRank()-1-i];

    //create the new numpy array
    PyObject *array = nullptr;
    if(copyflag)
    {
        array = PyArray_SimpleNewFromData(aptr->getRank(),
                                  dims,
                                  typeid2numpytc[tid],
                                  aptr->getStorage()->getStorage());
    }
    else
    {
        array = PyArray_SimpleNew(aptr->getRank(),dims,typeid2numpytc[tid]);
    }

    if(dims) delete [] dims;
    if(!array)
    {
        //THROW EXCEPTION HERE
        std::cerr<<"Error creating new numpy array!"<<std::endl;
    }

    handle<> h(array);

    return object(h);
}

