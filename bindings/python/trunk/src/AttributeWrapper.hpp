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
 * Created on: Jun 27, 2011
 *     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
 */

#ifndef __ATTRIBUTE_WRAPPER_HPP__
#define __ATTRIBUTE_WRAPPER_HPP__

#include <cdma/navigation/IAttribute.h>
#include "WrapperHelpers.hpp"

using namespace cdma;

/*! 
\brief Wrapps a CDAM attribute

This class wrapps a pointer to a CDMA attribute. It provides the IOObject
interface. 

*/
class AttributeWrapper
{
    private:
        IAttributePtr _ptr; //!< pointer to the attribute
    public:
        //!================public constructors=================================
        //! default constructor
        AttributeWrapper():_ptr(nullptr) {}

        //---------------------------------------------------------------------
        //! copy constructor
        AttributeWrapper(const AttributeWrapper &a): _ptr(a._ptr) {}

        //---------------------------------------------------------------------
        //! standard constructor
        AttributeWrapper(IAttributePtr ptr):_ptr(ptr) {}

        //---------------------------------------------------------------------
        //! destructor
        ~AttributeWrapper() {}

        //===================assignment operators==============================
        //! copy assignment operator
        AttributeWrapper &operator=(const AttributeWrapper &a)
        {
            if(this == &a) return *this;
            _ptr = a._ptr;
            return *this;
        }

        //---------------------------------------------------------------------
        //! return the data type of the attribute
        TypeID type() const;

        //--------------------------------------------------------------------
        //! return the shape of the attribute
        std::vector<size_t> shape() const;

        //---------------------------------------------------------------------
        //! return the rank of an attribute
        size_t rank() const;

        //---------------------------------------------------------------------
        //! return the size of the attribute
        size_t size() const { return _ptr->getLength(); }

        //---------------------------------------------------------------------
        //! return the name of the attribute
        std::string name() const { return _ptr->getName(); }

        //---------------------------------------------------------------------
        //! return attribute data
        template<typename T> T get() const { return 0;}

        ArrayWrapper get(const std::vector<size_t> &offset,
                         const std::vector<size_t> &shape) 
        { return ArrayWrapper();}

};

//========================implementation of template methods===================



#endif
