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
 * Created on: Jul 16, 2012
 *     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
 */

#include "TupleIterator.hpp"

/*! 
\brief PyIterator wrapper function

Templates function creates the Python type for an iterator for a particular
iterable type.
\param class_name name of the iterator class
*/
void wrap_tupleiterator()
{
    class_<TupleIterator>("TupleIterator")
        .def(init<>())
        .def("next",&TupleIterator::next)
        .def("__iter__",&TupleIterator::__iter__)
        ;
}

