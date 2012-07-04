#ifndef __ARRAYWRAPPER_HPP__
#define __ARRAYWRAPPER_HPP__

#include "Types.hpp"

class ArrayWrapper
{
    private:
        ArrayPtr _ptr; //!< pointer to the wrapped array object
    public:
        //================constructors and destructor==========================
        //! default constructor
        ArrayWrapper():_ptr(nullptr) {}

        //---------------------------------------------------------------------
        //! standard constructor
        ArrayWrapper(ArrayPtr ptr):_ptr(ptr) {}

        //---------------------------------------------------------------------
        //! copy constructor
        ArrayWrapper(const ArrayWrapper &a):_ptr(a._ptr) {}

        //---------------------------------------------------------------------
        //! destructor
        ~ArrayWrapper() {}

        //========================assignment operators=========================
        ArrayWrapper &operator=(const ArrayWrapper &a)
        {
            if(this == &a) return *this;
            _ptr = a._ptr;
            return *this;
        }

        //=====================================================================
        /*! 
        \brief get rank

        Return the number of dimensios of the array.
        \return number of dimension
        */
        size_t rank() const { return _ptr->getRank(); }

        //---------------------------------------------------------------------
        /*! 
        \brief get number of elements

        returns the number of elements stored in the array.
        \return array size
        */
        size_t size() const { return _ptr->getSize(); }

        //----------------------------------------------------------------------
        /*! 
        \brief get shape

        Return a vector with the number of elements along each dimension. 
        \return shape vector
        */
        std::vector<int> shape() const { return _ptr->getShape(); }

        //----------------------------------------------------------------------
        /*! 
        \brief get type ID

        Return the ID of the data type used to store the data.
        \return type ID
        */
        TypeID type() const 
        { 
            return typename2typeid[_ptr->getValueType().name()]; 
        }

        //----------------------------------------------------------------------
        //! read scalar data
        template<typename T> T get() const
        {
            return _ptr->getValue<T>();
        }

        //---------------------------------------------------------------------
        //! read from position
        template<typename T> T get(const std::vector<size_t> &pos) const
        {
            return _ptr->getValue<T>(pos);
        }

        //---------------------------------------------------------------------
        //! get pointer to data
        const void *ptr() const
        {
            return _ptr->getStorage()->getStorage();
        }
};

#endif
