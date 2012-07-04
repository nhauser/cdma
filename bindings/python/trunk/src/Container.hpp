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

template<typename TPTR> class ContainerWrapper
{
    private:
        TPTR _ptr; //! pointer to the container type
    protected:
        //==================protected constructors=============================
        //! standard constructor
        ContainerWrapper(TPTR ptr):_ptr(ptr) {}         
        
        //=============protected assignment operators==========================
        //! copy assignment operator
        ContainerWrapper &operator=(const ContainerWrapper &c)
        {
            if(this == &c) return *this;
            this->_ptr = c._ptr;
            return *this;
        }

        //===============protected members for child classes===================
        TPTR ptr() { return _ptr; }

        const TPTR ptr() const { return _ptr; }

    public:
        //============public constructors and destructor=======================
        //! default constructor
        ContainerWrapper():_ptr(nullptr) {}

        //---------------------------------------------------------------------
        //! destructor
        virtual ~ContainerWrapper() {}

        //==============member functions=======================================
        std::string location() const 
        { 
            return this->_ptr->getLocation(); 
        }

        std::string name() const 
        { 
            return this->_ptr->getName(); 
        }

        std::string short_name() const 
        { 
            return this->_ptr->getShortName(); 
        }

        bool is_group() const
        {
            if(_ptr->getContainerType() == IContainer::DATA_GROUP) 
                return true;
            else
                return false;
        }

};

template<typename TPTR> void wrap_container(const char* name)
{
    class_<ContainerWrapper<TPTR>>(name)
        .add_property("location",&ContainerWrapper<TPTR>::location)
        .add_property("name",&ContainerWrapper<TPTR>::name)
        .add_property("short_name",&ContainerWrapper<TPTR>::short_name)
        .add_property("is_group",&ContainerWrapper<TPTR>::is_group)
        ;
}



#endif
