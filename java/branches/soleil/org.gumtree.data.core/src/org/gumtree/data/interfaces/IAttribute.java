/*******************************************************************************
 * Copyright (c) 2008 Australian Nuclear Science and Technology Organisation.
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
 * CDMA Attribute, with name and value. The metadata for data items and groups.
 * @author nxi
 * 
 */
public interface IAttribute extends IModelObject {

  /**
   * Get the name of this Attribute. Attribute names are unique within a
   * NetcdfFile's global set, and within a Variable's set.
   * 
   * @return String object
   */
  String getName();

  /**
   * Get the data type of the Attribute value.
   * 
   * @return Class object
   */
  Class<?> getType();

  /**
   * True if value is a String or String[].
   * 
   * @return true or false
   */
  boolean isString();

  /**
   * True if value is an array (getLength() > 1).
   * 
   * @return true or false
   */
  boolean isArray();

  /**
   * Get the length of the array of values; = 1 if scaler.
   * 
   * @return integer value
   */
  int getLength();

  /**
   * Get the value as an Array.
   * 
   * @return Array of values.
   */
  IArray getValue();

  /**
   * Retrieve String value; only call if isString() is true.
   * 
   * @return String if this is a String valued attribute, else null.
   * @see IAttribute#isString
   */
  String getStringValue();

  /**
   * Retrieve String value; only call if isString() is true.
   * 
   * @param index integer value
   * @return String if this is a String valued attribute, else null.
   * @see IAttribute#isString
   */
  String getStringValue(int index);

  /**
   * Retrieve numeric value. Equivalent to <code>getNumericValue(0)</code>
   * 
   * @return the first element of the value array, or null if its a String.
   */
  Number getNumericValue();

  /**
   * Retrieve a numeric value by index. If its a String, it will try to parse
   * it as a double.
   * 
   * @param index the index into the value array.
   * @return Number <code>value[index]</code>, or null if its a non-parsable
   *         String or the index is out of range.
   */
  Number getNumericValue(int index);

  /**
   * Instances which have same content are equal.
   * 
   * @param o Object
   * @return true or false
   */
  boolean equals(Object o);

  /**
   * Override Object.hashCode() to implement equals.
   * 
   * @return integer value
   */
  int hashCode();

  /**
   * String representation.
   * 
   * @return String object
   */
  String toString();

  /**
   * set the value as a String, trimming trailing zeroes.
   * 
   * @param val String object
   */
  void setStringValue(String val);

  /**
   * set the values from an Array.
   * 
   * @param value IArray object
   */
  void setValue(IArray value);

}
