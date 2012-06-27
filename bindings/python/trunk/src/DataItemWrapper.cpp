#include "DataItemWrapper.hpp"

void init_array() { import_array(); }

//======================wrapper methods implementation=========================
std::string DataItemWrapper::_get_type_id(const std::type_info &tid) const
{
    if(tid.name() == typeid(int8_t).name()) return "int8";
    if(tid.name() == typeid(uint8_t).name()) return "uint8t";
    if(tid.name() == typeid(int16_t).name()) return "int16";
    if(tid.name() == typeid(uint16_t).name()) return "uint16";
    if(tid.name() == typeid(int32_t).name()) return "int32";
    if(tid.name() == typeid(uint32_t).name()) return "uint32";
    if(tid.name() == typeid(int64_t).name()) return "int64";
    if(tid.name() == typeid(uint64_t).name()) return "uint64";

    if(tid.name() == typeid(float).name()) return "float32";
    if(tid.name() == typeid(double).name()) return "float64";
    if(tid.name() == typeid(char).name()) return "string";

    //THROW AN EXCEPTION HERE
    std::cerr<<"Data type unsupported"<<std::endl;
    std::cerr<<tid.name()<<std::endl;
}

//-----------------------------------------------------------------------------
int DataItemWrapper::_get_type_num(const std::type_info &tid) const
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

    //THROW AN EXCEPTION HERE
    std::cerr<<"Data type unsupported"<<std::endl;
    std::cerr<<tid.name()<<std::endl;
}

//-----------------------------------------------------------------------------
tuple DataItemWrapper::shape() const
{
    list l;

    for(auto v: ptr()->getShape()) l.append(v);

    return tuple(l);
}

//-----------------------------------------------------------------------------
object DataItemWrapper::__getitem__(object selection) const
{
    //determine the data type of the object
    std::string tid = _get_type_id(ptr()->getType());
    init_array();

    if(ptr()->isScalar())
    {
        std::cout<<"Reading scalar data ..."<<std::endl;
        if(tid=="string") 
        {
            std::cout<<"Reading scalar string data ..."<<std::endl;
            std::cout<<(ptr()->readString())<<std::endl;
            return object(new std::string(ptr()->readString()));   
        }

        if((tid=="int8")||(tid=="uint8"))
            return object(ptr()->readScalarByte());

        if((tid=="int16")||(tid=="uint16"))
            return object(ptr()->readScalarShort());

        if((tid=="int32")||(tid=="uint32"))
            return object(ptr()->readScalarInt());
        
        if((tid=="int64")||(tid=="uint64"))
            return object(ptr()->readScalarLong());
    }
    else
    {
        //read data from the dataset
        std::cout<<"Reading data from file ..."<<std::endl;
        ArrayPtr aptr = ptr()->getData();
        //read image data
        npy_intp *dims = new npy_intp[aptr->getRank()]; //allocate memory for dimensions
        for(size_t i=0;i<aptr->getRank();i++) dims[i] = aptr->getShape()[i];

        std::cout<<"creating new numpy array ..."<<std::endl;
        //create the new numpy array
        PyObject *array = nullptr;
        std::cout<<_get_type_num(aptr->getValueType())<<std::endl;
        array = PyArray_SimpleNew(aptr->getRank(),
                                  dims,
                                  _get_type_num(aptr->getValueType()));
        std::cout<<"creating new numpy array ..."<<std::endl;
        if(!array)
        {
            std::cerr<<"Error creating new numpy array!"<<std::endl;
        }
        std::cout<<"creating object handler ..."<<std::endl;
        delete dims;
        handle<> h(array);

        std::cout<<"Create object ..."<<std::endl;
        return object(h);
    }

    return object(new std::string("hello world"));
}
//===============helper function creating the python class=====================
void wrap_dataitem()
{
    wrap_container<IDataItemPtr>("DataItemContainer");

    class_<DataItemWrapper,bases<ContainerWrapper<IDataItemPtr>> >("DataItem")
        .add_property("rank",&DataItemWrapper::rank)
        .add_property("shape",&DataItemWrapper::shape)
        .add_property("size",&DataItemWrapper::size)
        .add_property("unit",&DataItemWrapper::unit)
        .add_property("type",&DataItemWrapper::type)
        .def("__getitem__",&DataItemWrapper::__getitem__)
        ;
        
}
