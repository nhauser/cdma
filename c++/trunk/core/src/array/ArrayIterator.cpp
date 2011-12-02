//*****************************************************************************
// Synchrotron SOLEIL
//
// Creation : 08/12/2011
// Author   : Rodriguez Clément
//
// This program is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation; version 2 of the License.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
//
//*****************************************************************************

/**
 * Array iterator.
 * @author nxi
 *
 */

#include <cdma/exception/Exception.h>
#include <cdma/array/impl/ArrayIterator.h>
namespace cdma
{

	/**
	 * Return true if there are more elements in the iteration.
	 *
	 * @return true or false
	 */
	bool ArrayIterator::hasNext() { THROW_NOT_IMPLEMENTED("ArrayIterator::hasNext"); }

	/**
	 * Return true if there is an element in the current iteration.
	 *
	 * @return true or false
	 */
	bool ArrayIterator::hasCurrent() {THROW_NOT_IMPLEMENTED("ArrayIterator::hasCurrent"); }

	/**
	 * Get next value as a double.
	 *
	 * @return double value
	 */
	double ArrayIterator::getDoubleNext() {THROW_NOT_IMPLEMENTED("ArrayIterator::getDoubleNext"); }

	/**
	 * Set next value with a double.
	 *
	 * @param val
	 *            double value
	 */
	void ArrayIterator::setDoubleNext(double val) { }

	/**
	 * Get current value as a double.
	 *
	 * @return double value
	 */
	double ArrayIterator::getDoubleCurrent() {THROW_NOT_IMPLEMENTED("ArrayIterator::getDoubleCurrent"); }

	/**
	 * Set current value with a double.
	 *
	 * @param val
	 *            double value
	 */
	void ArrayIterator::setDoubleCurrent(double val) { }

	/**
	 * Get next value as a float.
	 *
	 * @return float value
	 */
	float ArrayIterator::getFloatNext() { THROW_NOT_IMPLEMENTED("ArrayIterator::getFloatNext"); }

	/**
	 * Set next value with a float.
	 *
	 * @param val
	 *            float value
	 */
	void ArrayIterator::setFloatNext(float val) { }

	/**
	 * Get current value as a float.
	 *
	 * @return float value
	 */
	float ArrayIterator::getFloatCurrent() {THROW_NOT_IMPLEMENTED("ArrayIterator::getFloatCurrent"); }

	/**
	 * Set current value with a float.
	 *
	 * @param val
	 *            float value
	 */
	void ArrayIterator::setFloatCurrent(float val) { }

	/**
	 * Get next value as a long.
	 *
	 * @return long value
	 */
	long ArrayIterator::getLongNext() { THROW_NOT_IMPLEMENTED("ArrayIterator::getLongNext"); }

	/**
	 * Set next value with a long.
	 *
	 * @param val
	 *            long value
	 */
	void ArrayIterator::setLongNext(long val) { }

	/**
	 * Get current value as a long.
	 *
	 * @return long value
	 */
	long ArrayIterator::getLongCurrent() { THROW_NOT_IMPLEMENTED("ArrayIterator::getLongCurrent"); }

	/**
	 * Set current value with a long.
	 *
	 * @param val
	 *            long value
	 */
	void ArrayIterator::setLongCurrent(long val) { }

	/**
	 * Get next value as a int.
	 *
	 * @return integer value
	 */
	int ArrayIterator::getIntNext() { THROW_NOT_IMPLEMENTED("ArrayIterator::"); }

	/**
	 * Set next value with a int.
	 *
	 * @param val
	 *            integer value
	 */
	void ArrayIterator::setIntNext(int val) { }

	/**
	 * Get current value as a int.
	 *
	 * @return integer value
	 */
	int ArrayIterator::getIntCurrent() { THROW_NOT_IMPLEMENTED("ArrayIterator::getIntCurrent"); }

	/**
	 * Set current value with a int.
	 *
	 * @param val
	 *            integer value
	 */
	void ArrayIterator::setIntCurrent(int val) { }

	/**
	 * Get next value as a short.
	 *
	 * @return short value
	 */
	short ArrayIterator::getShortNext() { THROW_NOT_IMPLEMENTED("ArrayIterator::getShortNext"); }

	/**
	 * Set next value with a short.
	 *
	 * @param val
	 *            short value
	 */
	void ArrayIterator::setShortNext(short val) { }

	/**
	 * Get current value as a short.
	 *
	 * @return short value
	 */
	short ArrayIterator::getShortCurrent() { THROW_NOT_IMPLEMENTED("ArrayIterator::getShortCurrent"); }

	/**
	 * Set current value with a short.
	 *
	 * @param val
	 *            short value
	 */
	void ArrayIterator::setShortCurrent(short val) { }

	/**
	 * Get next value as a byte.
	 *
	 * @return unsigned char value
	 */
	unsigned char ArrayIterator::getByteNext() { THROW_NOT_IMPLEMENTED("ArrayIterator::getByteNext"); }

	/**
	 * Set next value with a byte.
	 *
	 * @param val
	 *            unsigned char value
	 */
	void ArrayIterator::setByteNext(unsigned char val) { }

	/**
	 * Get current value as a byte.
	 *
	 * @return unsigned char value
	 */
	unsigned char ArrayIterator::getByteCurrent() { THROW_NOT_IMPLEMENTED("ArrayIterator::getByteCurrent"); }

	/**
	 * Set current value with a byte.
	 *
	 * @param val
	 *            unsigned char value
	 */
	void ArrayIterator::setByteCurrent(unsigned char val) { }

	/**
	 * Get next value as a char.
	 *
	 * @return char value
	 */
	char ArrayIterator::getCharNext() { THROW_NOT_IMPLEMENTED("ArrayIterator:getCharNext:"); }

	/**
	 * Set next value with a char.
	 *
	 * @param val
	 *            char value
	 */
	void ArrayIterator::setCharNext(char val) { }

	/**
	 * Get current value as a char.
	 *
	 * @return char value
	 */
	char ArrayIterator::getCharCurrent() { THROW_NOT_IMPLEMENTED("ArrayIterator::getCharCurrent"); }

	/**
	 * Set current value with a char.
	 *
	 * @param val
	 *            char value
	 */
	void ArrayIterator::setCharCurrent(char val) { }

	/**
	 * Get next value as a bool.
	 *
	 * @return true or false
	 */
	bool ArrayIterator::getBooleanNext() { THROW_NOT_IMPLEMENTED("ArrayIterator::getBooleanNext"); }

	/**
	 * Set next value with a bool.
	 *
	 * @param val
	 *            true or false
	 */
	void ArrayIterator::setBooleanNext(bool val) { }

	/**
	 * Get current value as a bool.
	 *
	 * @return true or false
	 */
	bool ArrayIterator::getBooleanCurrent() { THROW_NOT_IMPLEMENTED("ArrayIterator::getBooleanCurrent"); }

	/**
	 * Set current value with a bool.
	 *
	 * @param val
	 *            bool true or false
	 */
	void ArrayIterator::setBooleanCurrent(bool val) { }

	/**
	 * Get next value as an Object.
	 *
	 * @return Object
	 */
	yat::Any ArrayIterator::getObjectNext() { THROW_NOT_IMPLEMENTED("ArrayIterator::getObjectNext"); }

	/**
	 * Set next value with a Object.
	 *
	 * @param val
	 *            any Object
	 */
	void ArrayIterator::setObjectNext(const yat::Any& val) { }

	/**
	 * Get current value as a Object.
	 *
	 * @return Object
	 */
	yat::Any ArrayIterator::getObjectCurrent() { THROW_NOT_IMPLEMENTED("ArrayIterator::getObjectCurrent"); }

	/**
	 * Set current value with a Object.
	 *
	 * @param val
	 *            any Object
	 */
	void ArrayIterator::setObjectCurrent(const yat::Any& val) { }

	/**
	 * Get next value as an Object.
	 *
	 * @return any Object
	 */
	yat::Any ArrayIterator::next() { THROW_NOT_IMPLEMENTED("ArrayIterator::next"); }

	/**
	 * Get the current counter, use for debugging.
	 *
	 * @return array of integer
	 */
	std::vector<int> ArrayIterator::getCurrentCounter() { THROW_NOT_IMPLEMENTED("ArrayIterator::getCurrentCounter"); }
}