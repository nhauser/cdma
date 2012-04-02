/*******************************************************************************
 * Copyright (c) 2010 Australian Nuclear Science and Technology Organisation.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0 
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 * 
 * Contributors: 
 *    Norman Xiong (nxi@Bragg Institute) - initial API and implementation
 *    Clément Rodriguez (clement.rodriguez@synchrotron-soleil) - API evolution
 ******************************************************************************/
package org.gumtree.data.interfaces;

/**
 * Array iterator permits to iterate over an Array.
 * When initialized, it should be invalid: starting at index -1.
 * It means that hasNext() returns true and the first element is accessed
 * using get*Next().
 * The set methods replace the last element returned by <i>next<i> with the  
 * specified operation.
 * 
 * @author nxi
 * 
 */

public interface IArrayIterator extends IModelObject {

	/**
	 * Return true if there are more elements in the iteration.
	 * 
	 * @return true or false
	 */
	boolean hasNext();

	/**
	 * Get next value as a double.
	 * 
	 * @return double value
	 */
	double getDoubleNext();

	/**
	 * Set the value with a given double.
	 * 
	 * @param val double value
	 */
	void setDouble(double val);

	/**
	 * Get next value as a float.
	 * 
	 * @return float value
	 */
	float getFloatNext();

	/**
	 * Set the value with a float.
	 * 
	 * @param val float value
	 */
	void setFloat(float val);

	/**
	 * Get next value as a long.
	 * 
	 * @return long value
	 */
	long getLongNext();

	/**
	 * Set the value with a long.
	 * 
	 * @param val long value
	 */
	void setLong(long val);

	/**
	 * Get next value as a integer.
	 * 
	 * @return integer value
	 */
	int getIntNext();

	/**
	 * Set the value with a integer.
	 * 
	 * @param val integer value
	 */
	void setInt(int val);

	/**
	 * Get next value as a short.
	 * 
	 * @return short value
	 */
	short getShortNext();

	/**
	 * Set the value with a short.
	 * 
	 * @param val short value
	 */
	void setShort(short val);

	/**
	 * Get next value as a byte.
	 * 
	 * @return byte value
	 */
	byte getByteNext();

	/**
	 * Set the value with a byte.
	 * 
	 * @param val byte value
	 */
	void setByte(byte val);

	/**
	 * Get next value as a char.
	 * 
	 * @return char value
	 */
	char getCharNext();

	/**
	 * Set the value with a char.
	 * 
	 * @param val char value
	 */
	void setChar(char val);

	/**
	 * Get next value as a boolean.
	 * 
	 * @return true or false
	 */
	boolean getBooleanNext();

	/**
	 * Set the value with a boolean.
	 * 
	 * @param val true or false
	 */
	void setBoolean(boolean val);

	/**
	 * Get next value as an Object.
	 * 
	 * @return Object
	 */
	Object getObjectNext();

	/**
	 * Set the value with a Object.
	 * 
	 * @param val any Object
	 */
	void setObject(Object val);

	/**
	 * Get next value as an Object.
	 * 
	 * @return any Object
	 */
	Object next();

	/**
	 * Get the current counter, use for debugging.
	 * 
	 * @return array of integer
	 */
	int[] getCounter();
}
