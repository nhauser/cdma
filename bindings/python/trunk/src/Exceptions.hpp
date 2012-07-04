#ifndef __EXCEPTIONS_HPP__
#define __EXCEPTIONS_HPP__


//------------------------------------------------------------------------------
/*! 
\brief throw Python TypeError exception

Throw the TypeError Python exception.
*/
void throw_PyTypeError(const std::string &message);

//------------------------------------------------------------------------------
/*! 
\brief throw Python IndexError exception
*/
void throw_PyIndexError(const std::string &message);

//------------------------------------------------------------------------------
//! throw Python KeyError exception
void throw_PyKeyError(const std::string &message);

//------------------------------------------------------------------------------
template<typename ETYPE> void throw_cdma_exception(
        const std::string desc,
        const std::string origin)
{
    throw ETYPE(desc,origin);
}



#endif
