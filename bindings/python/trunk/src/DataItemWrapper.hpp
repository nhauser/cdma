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
        tuple shape() const;

        //---------------------------------------------------------------------
        size_t size() const { return ptr()->getSize(); }

        std::string unit() const { return ptr()->getUnitsString(); }

        std::string type() const { return get_type_string(ptr()->getType()); } 

        //---------------------------------------------------------------------
        object __getitem__(object selection) const;
        
        
};




#endif
