#ifndef __GROUPWRAPPER_HPP__
#define __GROUPWRAPPER_HPP__

#include<boost/python.hpp>
using namespace boost::python;

#include<cdma/navigation/IGroup.h>
using namespace cdma;

#include "Container.hpp"
#include "AttributeManager.hpp"

class GroupWrapper:public ContainerWrapper<IGroupPtr>
{
    public:
        //=====================public members==================================
        AttributeManager<IGroupPtr> attrs;
        //==================constructors and destructor========================
        //! default constructor
        GroupWrapper():ContainerWrapper<IGroupPtr>(),attrs() {}

        //---------------------------------------------------------------------
        //! standard constructor
        GroupWrapper(IGroupPtr g):ContainerWrapper<IGroupPtr>(g),attrs(g) {}

        //---------------------------------------------------------------------
        //! destructor
        ~GroupWrapper() {}

        //==================assignment operators===============================
        GroupWrapper &operator=(const GroupWrapper &g)
        {
            if(this == &g) return *this;
            ContainerWrapper<IGroupPtr>::operator=(g);
            attrs = g.attrs;
            return *this;
        }

        //===================data access methods===============================
        object __getitem__(const std::string &path) const;

        //---------------------------------------------------------------------
        //! returns a tuple of all childs of a group
        tuple childs() const;

        //---------------------------------------------------------------------
        //! returns a tuple of all groups below this group
        tuple groups() const;

        //---------------------------------------------------------------------
        //! returns a tuple of all data items below this group
        tuple items() const;

        //---------------------------------------------------------------------
        //! return the parent group 
        GroupWrapper parent() const;

        //---------------------------------------------------------------------
        //! return the root group
        GroupWrapper root() const;

        //---------------------------------------------------------------------
        /*!
        \brief return dimension list

        Returns a list of IDimensionPtr dimension pointers.
        \return dimension pointers
        */
        std::list<IDimensionPtr> dimensions() const;




};
        

#endif
