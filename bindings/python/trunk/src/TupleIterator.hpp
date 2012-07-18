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
 * Created on: Jul 16, 2012
 *     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
 */

#ifndef __TUPLEITERATOR_HPP__
#define __TUPLEITERATOR_HPP__

#include <boost/python.hpp>
#include <exception>

#include "Exceptions.hpp"

using namespace boost::python;

/*!
\ingroup utility_classes
\brief iterates over a tuple

The interesing aspect of this iterator is that it does not hold the reference to
a container object but the object itself. 
*/
class TupleIterator
{
    private:
        tuple _container; //!< parent object of the interator
        ssize_t   _state;          //!< position of the iterator
    public:
        //=======================constructors and destructor====================
        //! default constructor
        TupleIterator(): _container(), _state(0) {}   
        
        //---------------------------------------------------------------------
        //! copy constructor
        TupleIterator(const TupleIterator &i): 
            _container(i._container), 
            _state(i._state) 
        {}

        //---------------------------------------------------------------------
        /*! \brief construction from an ITERABLE instance

        \param i iterable object 
        \param state initial state of the iterator
        */
        explicit TupleIterator(const tuple &i,size_t state=0):
            _container(i),
            _state(state)
        { }

        //---------------------------------------------------------------------
        //! destructor
        ~TupleIterator()
        {
            _state  = 0;
        }

        //=======================assignment operators==========================
        //! copy assignment operator
        TupleIterator & operator=(const TupleIterator &i)
        {
            if(this != &i){
                _container = i._container;
                _state   = i._state;
            }
            return *this;
        }

        //---------------------------------------------------------------------
        /*! \brief return next element

        This method returns the next element of the container and increments the
        internal state of the iterator.
        \return object at actual iterator position
        */
        object next() 
        {
            //check if iteration is still possible
            if(_state == len(_container))
            {
                throw_PyStopIteration("iteration finished");
                return(object());
            }

            //return the current object 
            object item = _container[this->_state];
            //increment the iterator
            this->_state++;

            return item;
        }

        //----------------------------------------------------------------------
        //! \brief required by Python
        object __iter__()
        {
            return object(this);
        }

};

#endif
