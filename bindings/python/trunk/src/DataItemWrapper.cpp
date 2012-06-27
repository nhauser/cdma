#include "DataItemWrapper.hpp"


//======================wrapper methods implementation=========================
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
    std::string tid = get_type_string(ptr());
    init_numpy();

    if(ptr()->isScalar())
    {
        if(tid=="string") 
            return object(new std::string(ptr()->readString()));   

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
        std::vector<int> origin,shape;

        /*
        if(create_cdma_selection(selection,origin,shape))
        {
            //if returns true we have a point selection and thus return a scalar
            ArrayPtr aptr = ptr()->getData(origin);
            if((tid=="int8")||

        }
        else
        {*/
            //read data from the dataset
            ArrayPtr aptr = ptr()->getData();
            object array = cdma2numpy_array(aptr,true);
            return array;
        //}
    }
    
    //THROW EXCEPTION HERE
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
