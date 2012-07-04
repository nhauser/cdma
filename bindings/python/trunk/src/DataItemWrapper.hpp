#ifndef __DATAITEMWRAPPER_HPP__
#define __DATAITEMWRAPPER_HPP__

#include<cdma/navigation/IDataItem.h>

#include "Container.hpp"
#include "ArrayWrapper.hpp"
#include "WrapperHelpers.hpp"
#include "DimensionManager.hpp"


class DataItemWrapper:public ContainerWrapper<IDataItemPtr>
{
    public:
        //=======================public members================================
        DimensionManager dim; //!< dimension manager for this data item
        //================constructors and destructor==========================
        //! default constructor
        DataItemWrapper():ContainerWrapper<IDataItemPtr>(),dim() {}

        //---------------------------------------------------------------------
        //! standard constructor
        DataItemWrapper(IDataItemPtr item):
            ContainerWrapper<IDataItemPtr>(item),
            dim(item)
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
            dim = item.dim;
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
        \brief get description

        Return the description of the data item. 
        \return description string
        */
        std::string description() const { return ptr()->getDescription(); }

        //---------------------------------------------------------------------
        /*!
        \brief return type id

        Return the ID of the data type used to store data.
        \return type id
        */
        TypeID type() const;

        //---------------------------------------------------------------------
        /*!
        \brief returns a list of dimensions

        Returns a list of all dimensions associated with this data item. 
        \return list of dimensions
        */
        std::list<IDimensionPtr> dimensions() const;


        //---------------------------------------------------------------------
        template<typename T> T get() const { return 0; } 

        ArrayWrapper get(const std::vector<size_t> &offset,const
                std::vector<size_t> &
                shape);

        
        
};




#endif
