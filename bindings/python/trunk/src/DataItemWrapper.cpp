#include "DataItemWrapper.hpp"


//======================wrapper methods implementation=========================
std::vector<size_t> DataItemWrapper::shape() const
{
    std::vector<size_t> shape;

    for(auto v: ptr()->getShape()) shape.push_back(v);

    return shape;
}

//-----------------------------------------------------------------------------
TypeID DataItemWrapper::type() const
{
    return typename2typeid[ptr()->getType().name()];
}

//================overloaded scalar get template===============================
template<> uint8_t DataItemWrapper::get<uint8_t>() const
{
    return ptr()->readScalarByte();
}

//-----------------------------------------------------------------------------
template<> int16_t DataItemWrapper::get<int16_t>() const
{
    return ptr()->readScalarShort();
}

//-----------------------------------------------------------------------------
template<> int32_t DataItemWrapper::get<int32_t>() const
{
    return ptr()->readScalarInt();
}

//-----------------------------------------------------------------------------
template<> int64_t DataItemWrapper::get<int64_t>() const
{
    return ptr()->readScalarLong();
}

//-----------------------------------------------------------------------------
template<> float DataItemWrapper::get<float>() const
{
    return ptr()->readScalarFloat();
}

//-----------------------------------------------------------------------------
template<> double DataItemWrapper::get<double>() const
{
    return ptr()->readScalarDouble();
}

//-----------------------------------------------------------------------------
template<> std::string DataItemWrapper::get<std::string>() const
{
    return ptr()->readString();
}

//-----------------------------------------------------------------------------
/*
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

        if(create_cdma_selection(selection,origin,shape))
        {
            //if returns true we have a point selection and thus return a scalar
            ArrayPtr aptr = ptr()->getData(origin);
            if((tid=="int8")||

        }
        else
        {
            //read data from the dataset
            ArrayPtr aptr = ptr()->getData();
            object array = cdma2numpy_array(aptr,true);
            return array;
        //}
    }
    
    //THROW EXCEPTION HERE
}
*/
//===============helper function creating the python class=====================
void wrap_dataitem()
{
    wrap_container<IDataItemPtr>("DataItemContainer");

    class_<DataItemWrapper,bases<ContainerWrapper<IDataItemPtr>> >("DataItem")
        .add_property("rank",&DataItemWrapper::rank)
        .add_property("shape",&__shape__<DataItemWrapper>)
        .add_property("size",&DataItemWrapper::size)
        .add_property("unit",&DataItemWrapper::unit)
        .add_property("type",&__type__<DataItemWrapper>)
        .add_property("__getitem__",&__getitem__<DataItemWrapper>)
        ;
        
}
