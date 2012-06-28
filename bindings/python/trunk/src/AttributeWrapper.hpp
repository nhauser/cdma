#ifndef __ATTRIBUTE_WRAPPER_HPP__
#define __ATTRIBUTE_WRAPPER_HPP__

#include <cdma/navigation/IAttribute.h>
#include "WrapperHelpers.hpp"

using namespace cdma;

class AttributeWrapper
{
    private:
        IAttributePtr _ptr;
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
        //! return the size of the attribute
        size_t size() const { return _ptr->getLength(); }

        //---------------------------------------------------------------------
        //! return the name of the attribute
        std::string name() const { return _ptr->getName(); 
        }

        //---------------------------------------------------------------------
        //! return attribute data
        template<typename T> T get() const {}

};

//========================implementation of template methods===================



#endif
