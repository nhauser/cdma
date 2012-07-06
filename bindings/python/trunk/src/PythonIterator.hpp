#ifndef __PYTHONITERATOR_HPP__
#define __PYTHONITERATOR_HPP__

#include <boost/python.hpp>
#include <exception>

using namespace boost::python;


/*! 
\brief python compatible iterator

A simple forward iterator which can be used from within Python.
*/
template<typename ITERABLE> class PyIterator
{
    private:
        const ITERABLE *_container; //!< parent object of the interator
        size_t     _state;  //!< actual index 
                            //!< interator referes
    public:
        typedef typename ITERABLE::value_type value_type;    //!< type of the elements 
        typedef ITERABLE iterable_type; //!< type of the iterable
        //=======================constructors and destructor====================
        //! default constructor
        PyIterator():
            _container(nullptr),
            _state(0)
        {}   
        
        //---------------------------------------------------------------------
        //! copy constructor
        PyIterator(const PyIterator<ITERABLE> &i):
            _container(i._container),
            _state(i._state)
        {}

        //---------------------------------------------------------------------
        /*! \brief construction from an ITERABLE instance

        \param g iterable object 
        \param start_index index of the first element the iterator should point
        to
        */
        explicit PyIterator(const ITERABLE &i,size_t state=0):
            _container(&i),
            _state(state)
        { }

        //---------------------------------------------------------------------
        //! destructor
        ~PyIterator()
        {
            _container = nullptr;
            _state  = 0;
        }

        //=======================assignment operators==========================
        //! copy assignment operator
        PyIterator<PyIterator> & operator=(const PyIterator<ITERABLE> &i)
        {
            if(this != &i){
                _container = i._container;
                _state   = i._state;
            }
            return *this;
        }

        //---------------------------------------------------------------------
        /*! \brief return next element

        This method returns the next element of the container. 
        \return instance of ItemT with the next element
        */
        value_type next()
        {
            //check if iteration is still possible
            if(_state == this->_container->size())
            {
                throw_PyStopIteration("iteration finished");
                return(value_type());
            }

            //return the current object 
            value_type item((*(this->_container))[this->_state]);
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

//-----------------------------------------------------------------------------
/*! \brief AttributeIterator wrapper generator

This function creates the Python code for AttributeIterator objects. 
\param class_name name of the created Python class
*/
template<typename ITERABLE> void wrap_pyiterator(const std::string &class_name)
{
    class_<PyIterator<ITERABLE> >(class_name.c_str())
        .def(init<>())
        .def("next",&PyIterator<ITERABLE>::next)
        .def("__iter__",&PyIterator<ITERABLE>::__iter__)
        ;
}


#endif
