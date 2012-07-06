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
 * Created on: Jul 03, 2011
 *     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
 */

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
//! throw stop exception for iterators
void throw_PyStopIteration(const std::string &message);

//------------------------------------------------------------------------------
template<typename ETYPE> void throw_cdma_exception(
        const std::string desc,
        const std::string origin)
{
    throw ETYPE(desc,origin);
}



#endif
