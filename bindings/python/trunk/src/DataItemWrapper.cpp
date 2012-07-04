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
std::list<IDimensionPtr> DataItemWrapper::dimensions() const
{
    return ptr()->getDimensionList();
}

//-----------------------------------------------------------------------------
ArrayWrapper DataItemWrapper::get(const std::vector<size_t> &offset,
                                  const std::vector<size_t> & shape)
{
    std::vector<int> _offset(offset.size());
    std::vector<int> _shape(shape.size());
    size_t index = 0;
    for(auto &v: offset) _offset[index++] = v;
    index = 0;
    for(auto &v: shape) _shape[index++] = v;
    return ArrayWrapper(ptr()->getData(_offset,_shape));
}

//===============helper function creating the python class=====================
void wrap_dataitem()
{
    wrap_container<IDataItemPtr>("DataItemContainer");
    wrap_dimensionmanager();

    class_<DataItemWrapper,bases<ContainerWrapper<IDataItemPtr>> >("DataItem")
        .def_readwrite("dim",&DataItemWrapper::dim)
        .add_property("rank",&DataItemWrapper::rank)
        .add_property("shape",&__shape__<DataItemWrapper>)
        .add_property("size",&DataItemWrapper::size)
        .add_property("unit",&DataItemWrapper::unit)
        .add_property("type",&__type__<DataItemWrapper>)
        .add_property("description",&DataItemWrapper::description)
        .add_property("dims",&__dimensions__<DataItemWrapper>)
        .def("__getitem__",&__getitem__<DataItemWrapper>)
        ;
        
}
