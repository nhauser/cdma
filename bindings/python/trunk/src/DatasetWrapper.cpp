
#include "DatasetWrapper.hpp"
#include "GroupWrapper.hpp"
#include "DataItemWrapper.hpp"




//==============help function creating the python class=======================
void wrap_dataset()
{
    class_<DatasetWrapper>("Dataset")
        .def(init<>())
        .add_property("title",&DatasetWrapper::getTitle)
        .add_property("location",&DatasetWrapper::getLocation)
        .add_property("childs",&DatasetWrapper::childs)
        .def("__getitem__",&DatasetWrapper::__getitem__)
        ;
}


