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
//[SOLEIL][clement][11/24/2011] it appears important, to me, to fit with the behavior of standard java.util.Iterator .
// To extend it isn't mandatory (it will even 'break your framework' because of parameterized types) but to respect
// the convention will ease its use and will prevent to have different usage between to plugin

public interface IArrayIterator extends IModelObject {

	/**
	 * Return true if there are more elements in the iteration.
	 * 
	 * @return true or false
	 */
	boolean hasNext();

	/**
	 * Return true if there is an element in the current iteration.
	 * 
	 * @return true or false
	 * @deprecated
	 */
	boolean hasCurrent();
	
	/**
	 * Get next value as a double.
	 * 
	 * @return double value
	 */
	double getDoubleNext();

	/**
	 * Set next value with a double.
	 * 
	 * @param val
	 *            double value
	 * @deprecated prefer using IArrayIterator.setDouble()
	 */
	void setDoubleNext(double val);
	void setDouble(double val);

	/**
	 * Get current value as a double.
	 * 
	 * @return double value
	 * @deprecated
	 */
	double getDoubleCurrent();

	/**
	 * Set current value with a double.
	 * 
	 * @param val
	 *            double value
	 * @deprecated prefer using IArrayIterator.setDouble()
	 */
	void setDoubleCurrent(double val);

	/**
	 * Get next value as a float.
	 * 
	 * @return float value
	 */
	float getFloatNext();

	/**
	 * Set next value with a float.
	 * 
	 * @param val
	 *            float value
	 * @deprecated prefer using IArrayIterator.setFloat()
	 */
	void setFloatNext(float val);

	/**
	 * Get current value as a float.
	 * 
	 * @return float value
	 * @deprecated
	 */
	float getFloatCurrent();

	/**
	 * Set current value with a float.
	 * 
	 * @param val
	 *            float value
	 * @deprecated prefer using IArrayIterator.setFloat()
	 */
	void setFloatCurrent(float val);
	void setFloat(float val);

	/**
	 * Get next value as a long.
	 * 
	 * @return long value
	 */
	long getLongNext();

	/**
	 * Set next value with a long.
	 * 
	 * @param val
	 *            long value
	 * @deprecated prefer using IArrayIterator.setLong()
	 */
	void setLongNext(long val);

	/**
	 * Get current value as a long.
	 * 
	 * @return long value
	 * @deprecated
	 */
	long getLongCurrent();

	/**
	 * Set current value with a long.
	 * 
	 * @param val
	 *            long value
	 * @deprecated prefer using IArrayIterator.setLong()
	 */
	void setLongCurrent(long val);
	void setLong(long val);

	/**
	 * Get next value as a int.
	 * 
	 * @return integer value
	 */
	int getIntNext();

	/**
	 * Set next value with a int.
	 * 
	 * @param val
	 *            integer value
	 * @deprecated prefer using IArrayIterator.setInt()
	 */
	void setIntNext(int val);
	void setInt(int val);

	/**
	 * Get current value as a int.
	 * 
	 * @return integer value
	 * @deprecated prefer using IArrayIterator.getIntNext()
	 */
	int getIntCurrent();

	/**
	 * Set current value with a int.
	 * 
	 * @param val
	 *            integer value
	 * @deprecated prefer using IArrayIterator.setInt()
	 */
	void setIntCurrent(int val);

	/**
	 * Get next value as a short.
	 * 
	 * @return short value
	 */
	short getShortNext();

	/**
	 * Set next value with a short.
	 * 
	 * @param val
	 *            short value
	 * @deprecated prefer using IArrayIterator.setShort()
	 */
	void setShortNext(short val);

	/**
	 * Get current value as a short.
	 * 
	 * @return short value
	 * @deprecated prefer using IArrayIterator.getShort()
	 */
	short getShortCurrent();
	short getShort();

	/**
	 * Set current value with a short.
	 * 
	 * @param val
	 *            short value
	 * @deprecated prefer using IArrayIterator.setShort()
	 */
	void setShortCurrent(short val);
	void setShort(short val);

	/**
	 * Get next value as a byte.
	 * 
	 * @return byte value
	 */
	byte getByteNext();

	/**
	 * Set next value with a byte.
	 * 
	 * @param val
	 *            byte value
	 * @deprecated prefer using IArrayIterator.setByte()
	 */
	void setByteNext(byte val);
	void setByte(byte val);

	/**
	 * Get current value as a byte.
	 * 
	 * @return byte value
	 * @deprecated prefer using IArrayIterator.getByteNext()
	 */
	byte getByteCurrent();

	/**
	 * Set current value with a byte.
	 * 
	 * @param val
	 *            byte value
	 * @deprecated prefer using IArrayIterator.setByte()
	 */
	void setByteCurrent(byte val);

	/**
	 * Get next value as a char.
	 * 
	 * @return char value
	 */
	char getCharNext();

	/**
	 * Set next value with a char.
	 * 
	 * @param val
	 *            char value
	 * @deprecated prefer using IArrayIterator.setChar()
	 */
	void setCharNext(char val);
	void setChar(char val);

	/**
	 * Get current value as a char.
	 * 
	 * @return char value
	 * @deprecated prefer using IArrayIterator.getCharNext()
	 */
	char getCharCurrent();

	/**
	 * Set current value with a char.
	 * 
	 * @param val
	 *            char value
	 * @deprecated prefer using IArrayIterator.setChar()
	 */
	void setCharCurrent(char val);

	/**
	 * Get next value as a boolean.
	 * 
	 * @return true or false
	 */
	boolean getBooleanNext();

	/**
	 * Set next value with a boolean.
	 * 
	 * @param val
	 *            true or false
	 * @deprecated prefer using IArrayIterator.setBoolean()
	 */
	void setBooleanNext(boolean val);
	void setBoolean(boolean val);

	/**
	 * Get current value as a boolean.
	 * 
	 * @return true or false
	 * @deprecated prefer using IArrayIterator.getBooleanNext()
	 */
	boolean getBooleanCurrent();

	/**
	 * Set current value with a boolean.
	 * 
	 * @param val
	 *            boolean true or false
	 * @deprecated prefer using IArrayIterator.setBoolean()
	 */
	void setBooleanCurrent(boolean val);

	/**
	 * Get next value as an Object.
	 * 
	 * @return Object
	 */
	Object getObjectNext();

	/**
	 * Set next value with a Object.
	 * 
	 * @param val
	 *            any Object
	 * @deprecated prefer using IArrayIterator.setObject()
	 */
	void setObjectNext(Object val);
	void setObject(Object val);

	/**
	 * Get current value as a Object.
	 * 
	 * @return Object
	 * @deprecated prefer using IArrayIterator.getObjectNext()
	 */
	Object getObjectCurrent();

	/**
	 * Set current value with a Object.
	 * 
	 * @param val
	 *            any Object
	 * @deprecated prefer using IArrayIterator.setObject()
	 */
	void setObjectCurrent(Object val);

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
	 * @deprecated prefer using IArrayIterator.getCounter()
	 */
	int[] getCurrentCounter();
	int[] getCounter();
}
