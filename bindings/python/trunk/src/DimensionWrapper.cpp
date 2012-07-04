#include "DimensionWrapper.hpp"
#include "ArrayWrapper.hpp"
#include "WrapperHelpers.hpp"

//-----------------------------------------------------------------------------
object DimensionWrapper::axis() const
{
    ArrayWrapper array(_ptr->getCoordinateVariable());
    return cdma2numpy_array(array,true);
}

//-----------------------------------------------------------------------------
void wrap_dimension()
{
    class_<DimensionWrapper>("Dimension")
        .add_property("name",&DimensionWrapper::name)
        .add_property("size",&DimensionWrapper::size)
        .add_property("dim",&DimensionWrapper::dim)
        .add_property("order",&DimensionWrapper::order)
        .add_property("unit",&DimensionWrapper::unit)
        .add_property("axis",&DimensionWrapper::axis)
        ;
}
