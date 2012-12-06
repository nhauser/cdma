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
 * Created on: Jul 04, 2011
 *     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
 */

#ifndef __DIMENSIONWRAPPER_HPP__
#define __DIMENSIONWRAPPER_HPP__

#include<cdma/navigation/IDimension.h>
#include<boost/python.hpp>

using namespace cdma;
using namespace boost::python;

/*! 
\ingroup wrapper_classes
\brief wraps IDimensionPtr

Wraps a pointer of type IDimensionPtr.
*/
class DimensionWrapper
{
    private:
        IDimensionPtr _ptr; //!< pointer to a dimension
    public:
        //============constructors and destructor==============================
        //! default constructor
        DimensionWrapper():_ptr(nullptr) {}

        //---------------------------------------------------------------------
        //! copy constructor
        DimensionWrapper(const DimensionWrapper &d):_ptr(d._ptr) {}

        //---------------------------------------------------------------------
        //! standard constructor
        DimensionWrapper(IDimensionPtr ptr):_ptr(ptr) {}

        //---------------------------------------------------------------------
        //! destructor
        ~DimensionWrapper() {}

        //=================assignment operator=================================
        //! copy assignment operator
        DimensionWrapper &operator=(const DimensionWrapper &o)
        {
            if(this == &o) return *this;
            _ptr = o._ptr;
            return *this;
        }

        //=====================================================================
        /*! 
        \brief get name

        Return the name of the dimension
        \return name as string
        */
        std::string name() const { return _ptr->getName(); }

        //---------------------------------------------------------------------
        /*! 
        \brief get size

        Return the size (number of elements) of the dimension.
        \return dimension size
        */
        size_t size() const { return _ptr->getSize(); }

        //---------------------------------------------------------------------
        /*! 
        \brief get dimension

        Get the index of the dimension this dimension belongs to. 
        \return index of dimension
        */
        size_t dim() const { return _ptr->getDimensionAxis(); }

        //---------------------------------------------------------------------
        /*! 
        \brief order

        Get index of this dimension if several dimension are assiociated with
        a particular dimension of a DataItem. 
        \return dimension order
        */
        size_t order() const { return _ptr->getDisplayOrder(); }

        //---------------------------------------------------------------------
        /*! 
        \brief get unit

        Return the unit of the dimension as string. 
        \return unit string
        */
        std::string unit() const { return _ptr->getUnit(); }

        //----------------------------------------------------------------------
        /*! 
        \brief get axis

        Return the axis values for this dimension. This method returns a numpy
        array with the axis values.
        \return axis
        */
        object axis() const;

};


#endif
