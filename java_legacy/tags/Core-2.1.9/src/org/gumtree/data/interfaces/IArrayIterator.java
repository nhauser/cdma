/*******************************************************************************
 * Copyright (c) 2010 Australian Nuclear Science and Technology Organisation.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0 
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 * 
 * Contributors: 
 *    Norman Xiong (nxi@Bragg Institute) - initial API and implementation
 ******************************************************************************/
package org.gumtree.data.interfaces;

/**
 * Array iterator.
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
	 * Return true if there is an element in the current iteration.
	 * 
	 * @return true or false
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
	 */
	void setDoubleNext(double val);

	/**
	 * Get current value as a double.
	 * 
	 * @return double value
	 */
	double getDoubleCurrent();

	/**
	 * Set current value with a double.
	 * 
	 * @param val
	 *            double value
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
	 */
	void setFloatNext(float val);

	/**
	 * Get current value as a float.
	 * 
	 * @return float value
	 */
	float getFloatCurrent();

	/**
	 * Set current value with a float.
	 * 
	 * @param val
	 *            float value
	 */
	void setFloatCurrent(float val);

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
	 */
	void setLongNext(long val);

	/**
	 * Get current value as a long.
	 * 
	 * @return long value
	 */
	long getLongCurrent();

	/**
	 * Set current value with a long.
	 * 
	 * @param val
	 *            long value
	 */
	void setLongCurrent(long val);

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
	 */
	void setIntNext(int val);

	/**
	 * Get current value as a int.
	 * 
	 * @return integer value
	 */
	int getIntCurrent();

	/**
	 * Set current value with a int.
	 * 
	 * @param val
	 *            integer value
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
	 */
	void setShortNext(short val);

	/**
	 * Get current value as a short.
	 * 
	 * @return short value
	 */
	short getShortCurrent();

	/**
	 * Set current value with a short.
	 * 
	 * @param val
	 *            short value
	 */
	void setShortCurrent(short val);

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
	 */
	void setByteNext(byte val);

	/**
	 * Get current value as a byte.
	 * 
	 * @return byte value
	 */
	byte getByteCurrent();

	/**
	 * Set current value with a byte.
	 * 
	 * @param val
	 *            byte value
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
	 */
	void setCharNext(char val);

	/**
	 * Get current value as a char.
	 * 
	 * @return char value
	 */
	char getCharCurrent();

	/**
	 * Set current value with a char.
	 * 
	 * @param val
	 *            char value
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
	 */
	void setBooleanNext(boolean val);

	/**
	 * Get current value as a boolean.
	 * 
	 * @return true or false
	 */
	boolean getBooleanCurrent();

	/**
	 * Set current value with a boolean.
	 * 
	 * @param val
	 *            boolean true or false
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
	 */
	void setObjectNext(Object val);

	/**
	 * Get current value as a Object.
	 * 
	 * @return Object
	 */
	Object getObjectCurrent();

	/**
	 * Set current value with a Object.
	 * 
	 * @param val
	 *            any Object
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
	 */
	int[] getCurrentCounter();
}
