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
#ifndef __CONTAINER_HPP__
#define __CONTAINER_HPP__

#include<boost/python.hpp>
#include<cdma/navigation/IContainer.h>

using namespace boost::python;
using namespace cdma;

#include "AttributeManager.hpp"

/*! 
\brief wrapper for IContainer interface

Wraps objects that satisfy the IContainer interface. 
*/
template<typename TPTR> class ContainerWrapper
{
    private:
        TPTR _ptr; //! pointer to the container type
    protected:
        //==================protected constructors=============================
        //! standard constructor
        ContainerWrapper(TPTR ptr):_ptr(ptr),attrs(ptr) {}         
        
        //=============protected assignment operators==========================
        //! copy assignment operator
        ContainerWrapper &operator=(const ContainerWrapper &c)
        {
            if(this == &c) return *this;
            this->_ptr = c._ptr;
            this->attrs = c.attrs;
            return *this;
        }

        //===============protected members for child classes===================
        //! get a pointer to the container
        TPTR ptr() { return _ptr; }

        //! get a const pointer to the container
        const TPTR ptr() const { return _ptr; }

    public:
        //=================public members======================================
        AttributeManager<TPTR> attrs; //!< attribute manager

        //============public constructors and destructor=======================
        //! default constructor
        ContainerWrapper():_ptr(nullptr),attrs(nullptr) {}

        //---------------------------------------------------------------------
        //! destructor
        virtual ~ContainerWrapper() {}

        //==============member functions=======================================
        /*! 
        \brief returns object location

        This returns the path of the parent group of an object.
        \return path of the parent node
        */
        std::string location() const 
        { 
            return this->_ptr->getLocation(); 
        }

        //---------------------------------------------------------------------
        /*! 
        \brief full object name

        Returns the full path to the object.
        \return full path
        */
        std::string name() const 
        { 
            return this->_ptr->getName(); 
        }

        //---------------------------------------------------------------------
        /*! 
        \brief object name

        Returns the name of an object.
        \return object name
        */
        std::string short_name() const 
        { 
            return this->_ptr->getShortName(); 
        }

        //---------------------------------------------------------------------
        //! returns true if object is a group
        bool is_group() const
        {
            if(_ptr->getContainerType() == IContainer::DATA_GROUP) 
                return true;
            else
                return false;
        }

};

//=============================================================================
static const char __container_doc_attrs [] =
"Reference to the attribute manager of this container type";
static const char __container_doc_location [] = 
"location of this container in the tree";
static const char __container_doc_name [] = 
"name of the container type";
static const char __container_doc_short_name [] = 
"short name of the container";
static const char __container_doc_is_group [] = 
"attribute which is true of the container object is a container, false otherwise";

template<typename TPTR> void wrap_container(const char* name)
{
    class_<ContainerWrapper<TPTR>>(name)
        .def_readwrite("attrs",&ContainerWrapper<TPTR>::attrs,
                       __container_doc_attrs)
        .add_property("location",&ContainerWrapper<TPTR>::location,
                      __container_doc_location)
        .add_property("name",&ContainerWrapper<TPTR>::name,
                      __container_doc_name)
        .add_property("short_name",&ContainerWrapper<TPTR>::short_name,
                      __container_doc_short_name)
        .add_property("is_group",&ContainerWrapper<TPTR>::is_group,
                      __container_doc_is_group)
        ;
}



#endif
