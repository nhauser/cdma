#ifndef __DIMENSIONWRAPPER_HPP__
#define __DIMENSIONWRAPPER_HPP__

#include<cdma/navigation/IDimension.h>
#include<boost/python.hpp>

using namespace cdma;
using namespace boost::python;

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
        size_t size() const { return _ptr->getLength(); }

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
        std::string unit() const { return _ptr->getUnitsString(); }

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
