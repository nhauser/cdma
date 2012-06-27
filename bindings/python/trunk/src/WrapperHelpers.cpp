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
std::string get_type_string(ArrayPtr ptr) 
{
    return numpytc2numpystr[typename2numpytc[ptr->getValueType().name()]];
}

//-----------------------------------------------------------------------------
int get_type_code(ArrayPtr ptr) 
{
    return typename2numpytc[ptr->getValueType().name()];
}

//-----------------------------------------------------------------------------
size_t get_type_size(ArrayPtr ptr)
{
    return numpytc2size[typename2numpytc[ptr->getValueType().name()]];
}

//-----------------------------------------------------------------------------
object cdma2numpy_array(const ArrayPtr aptr,bool copyflag)
{
    init_numpy();
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
                                  get_type_code(aptr),
                                  aptr->getStorage()->getStorage());
    }
    else
    {
        array = PyArray_SimpleNew(aptr->getRank(),dims,get_type_code(aptr));
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

//-----------------------------------------------------------------------------
void copy_data_from_cdma2numpy(const ArrayPtr aptr,object &nparray)
{
    init_numpy();
    char *cdma_ptr = (char *)aptr->getStorage()->getStorage();
    char *nump_ptr = PyArray_BYTES(nparray.ptr());
    size_t esize = get_type_size(aptr);

    for(size_t i=0;i<aptr->getSize();i++)
        nump_ptr[i*esize] = cdma_ptr[i*esize];

}
