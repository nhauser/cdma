#include "AttributeWrapper.hpp"

//-----------------------------------------------------------------------------
TypeID AttributeWrapper::type() const
{
    if(_ptr->isString()) return TypeID::STRING;
    
    return typename2typeid[_ptr->getType().name()];
}

//-----------------------------------------------------------------------------
std::vector<size_t> AttributeWrapper::shape() const
{
    std::vector<size_t> shape;

    return shape;
}

//------------------------------------------------------------------------------
template<> float AttributeWrapper::get<float>() const
{
    return _ptr->getFloatValue();
}

//------------------------------------------------------------------------------
template<> int AttributeWrapper::get<int>() const
{
    return _ptr->getIntValue();
}

//------------------------------------------------------------------------------
template<> std::string AttributeWrapper::get<std::string>() const
{
    return _ptr->getStringValue();
}

//========================wrap attribute objects===============================
void wrap_attribute()
{
    class_<AttributeWrapper>("Attribute")
        .add_property("size",&AttributeWrapper::size)
        .add_property("name",&AttributeWrapper::name)
        .add_property("type",&__type__<AttributeWrapper>)
        .def("__getitem__",&__getitem__<AttributeWrapper>)
        ;
}
