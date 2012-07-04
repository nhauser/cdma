#ifndef __ATTRIBUTE_MANAGER_HPP__
#define __ATTRIBUTE_MANAGER_HPP__

#include<cdma/navigation/IContainer.h>
#include "AttributeWrapper.hpp"

using namespace cdma;

template<typename CPTR> class AttributeManager
{
    private:
        CPTR _ptr; //!< pointer to the container object
    public:
        //===================public constructors===============================
        //! default constructor
        AttributeManager():_ptr(nullptr) {}

        //---------------------------------------------------------------------
        //! copy constructor
        AttributeManager(const AttributeManager<CPTR> &m):_ptr(m._ptr) 
        { } 

        //---------------------------------------------------------------------
        //! default constructor
        AttributeManager(CPTR ptr):_ptr(ptr) {}

        //---------------------------------------------------------------------
        //! destructor
        ~AttributeManager() {}

        //====================assignment operators=============================
        //! copy assignment operator
        AttributeManager<CPTR> &operator=(const AttributeManager<CPTR> &m)
        {
            if(this == &m) return *this;
            _ptr = m._ptr;
            return *this;
        }

        //====================attribute related methods========================
        AttributeWrapper __getitem__str(const std::string &name) const 
        {
            IAttributePtr ptr = nullptr;
            ptr = this->_ptr->getAttribute(name);
            if(!ptr)
                throw_PyKeyError("Attribute ["+name+"] not found!");

            return AttributeWrapper(this->_ptr->getAttribute(name));
        }

        //---------------------------------------------------------------------
        size_t __len__() const
        {
            return this->_ptr->getAttributeList().size();
        }

        
};

//===================implementation of template methods========================
template<typename CPTR> void wrap_attribute_manager(const char *name)
{
    class_<AttributeManager<CPTR>>(name)
        .def("__getitem__",&AttributeManager<CPTR>::__getitem__str)
        .def("__len__",&AttributeManager<CPTR>::__len__)
        ;
}

#endif
