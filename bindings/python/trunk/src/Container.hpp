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
