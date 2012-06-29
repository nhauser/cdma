#ifndef __DATAITEMWRAPPER_HPP__
#define __DATAITEMWRAPPER_HPP__

#include<cdma/navigation/IDataItem.h>

#include "Container.hpp"
#include "WrapperHelpers.hpp"


class DataItemWrapper:public ContainerWrapper<IDataItemPtr>
{
    public:
        //================constructors and destructor==========================
        //! default constructor
        DataItemWrapper():ContainerWrapper<IDataItemPtr>() {}

        //---------------------------------------------------------------------
        //! standard constructor
        DataItemWrapper(IDataItemPtr item):ContainerWrapper<IDataItemPtr>(item)
        {}

        //---------------------------------------------------------------------
        //! destructor
        ~DataItemWrapper() {}

        //====================assignment operator==============================
        //! copy assignment operator
        DataItemWrapper &operator=(const DataItemWrapper &item)
        {
            if(this == &item) return *this;
            ContainerWrapper<IDataItemPtr>::operator=(item);
            return *this;
        }
        
        //======================inquery methods================================
        /*! 
        \brief get number of dimensions

        Return the number of dimensions from a DataItem object.
        \return number of dimensions
        */
        size_t rank() const { return ptr()->getRank(); }

        //---------------------------------------------------------------------
        /*! 
        \brief return data item shape

        Return the shape of a data item as a Python list. Each entry describes
        the number of elements along this particular dimension.
        \return Python list with number of elements
        */
        std::vector<size_t> shape() const;

        //---------------------------------------------------------------------
        /*! 
        \brief return number of elements

        Return the numbner of elements stored in the data item.
        \return number of elements
        */
        size_t size() const { return ptr()->getSize(); }

        //---------------------------------------------------------------------
        /*! 
        \brief return unit

        Return the unit of the data stored in this item as string.
        \return unit 
        */
        std::string unit() const { return ptr()->getUnitsString(); }

        //---------------------------------------------------------------------
        /*!
        \brief return type id

        Return the ID of the data type used to store data.
        \return type id
        */
        TypeID type() const;

        //---------------------------------------------------------------------
        template<typename T> T get() const {} 

        
        
};




#endif
