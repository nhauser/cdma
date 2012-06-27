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
std::string get_type_string(const std::type_info &tid) 
{
    switch(get_type_code(tid))
    {
        case NPY_BYTE: return "int8"; 
        case NPY_UBYTE: return "uint8";
        case NPY_SHORT: return "int16";
        case NPY_USHORT: return "uint16";
        case NPY_INT: return "int32";
        case NPY_UINT: return "uint32";
        case NPY_LONG: return "int64";
        case NPY_ULONG: return "uint64";
        case NPY_FLOAT: return "float32";
        case NPY_DOUBLE: return "float64";
        default:
            throw_PyTypeError("No type string available for this type!");
    };

}

//-----------------------------------------------------------------------------
int get_type_code(const std::type_info &tid) 
{
    if(tid.name() == typeid(int8_t).name()) return NPY_BYTE;
    if(tid.name() == typeid(uint8_t).name()) return NPY_UBYTE;
    if(tid.name() == typeid(int16_t).name()) return NPY_SHORT;
    if(tid.name() == typeid(uint16_t).name()) return NPY_USHORT;
    if(tid.name() == typeid(int32_t).name()) return NPY_INT;
    if(tid.name() == typeid(uint32_t).name()) return NPY_UINT;
    if(tid.name() == typeid(int64_t).name()) return NPY_LONG;
    if(tid.name() == typeid(uint64_t).name()) return NPY_ULONG;

    if(tid.name() == typeid(float).name()) return NPY_FLOAT;
    if(tid.name() == typeid(double).name()) return NPY_DOUBLE;

    throw_PyTypeError("Data is of unknown type!");
}

//-----------------------------------------------------------------------------
size_t get_type_size(const std::type_info &t)
{
    switch(get_type_code(t))
    {
        case NPY_BYTE: return sizeof(int8_t); 
        case NPY_UBYTE: return sizeof(uint8_t);
        case NPY_SHORT: return sizeof(int16_t);
        case NPY_USHORT: return sizeof(uint16_t);
        case NPY_INT: return sizeof(int32_t);
        case NPY_UINT: return sizeof(uint32_t);
        case NPY_LONG: return sizeof(int64_t);
        case NPY_ULONG: return sizeof(uint64_t);
        case NPY_FLOAT: return sizeof(float);
        case NPY_DOUBLE: return sizeof(double);
        default:
            throw_PyTypeError("Cannot determine size of this type!");
    };
}

//-----------------------------------------------------------------------------
object cdma2numpy_array(const ArrayPtr aptr)
{
    init_numpy();
    //set the dimension of the new array
    npy_intp *dims = new npy_intp[aptr->getRank()]; 
    for(size_t i=0;i<aptr->getRank();i++) 
        dims[i] = aptr->getShape()[aptr->getRank()-1-i];

    //create the new numpy array
    PyObject *array = nullptr;
    array = PyArray_SimpleNew(aptr->getRank(),
                              dims,
                              get_type_code(aptr->getValueType()));
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
    size_t esize = get_type_size(aptr->getValueType());

    for(size_t i=0;i<aptr->getSize();i++)
        nump_ptr[i*esize] = cdma_ptr[i*esize];

}
