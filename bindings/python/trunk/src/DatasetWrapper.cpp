/*
 * (c) Copyright 2012 DESY, Eugen Wintersberger <eugen.wintersberger@desy.de>
 *
 * This file is part of cdma-python.
 *
 * cdma-python is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 *
 * cdma-python is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with cmda-python.  If not, see <http://www.gnu.org/licenses/>.
 *************************************************************************
 *
 * Created on: Jun 26, 2011
 *     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
 */

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


