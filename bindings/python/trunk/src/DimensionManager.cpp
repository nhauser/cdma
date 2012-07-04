#include "DimensionManager.hpp"

//-----------------------------------------------------------------------------
void wrap_dimensionmanager()
{
    class_<DimensionManager>("DimensionManager")
        .def("__len__",&DimensionManager::size)
        .def("__getitem__",&DimensionManager::dimensions)
        ;
}
