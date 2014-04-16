/*******************************************************************************
 * Copyright (c) 2008 - ANSTO/Synchrotron SOLEIL
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 * 
 * Contributors:
 * 	Norman Xiong (nxi@Bragg Institute) - initial API and implementation
 * 	Tony Lam (nxi@Bragg Institute) - initial API and implementation
 *        Majid Ounsy (SOLEIL Synchrotron) - API v2 design and conception
 *        Stéphane Poirier (SOLEIL Synchrotron) - API v2 design and conception
 * 	Clement Rodriguez (ALTEN for SOLEIL Synchrotron) - API evolution
 * 	Gregory VIGUIER (SOLEIL Synchrotron) - API evolution
 ******************************************************************************/
// ****************************************************************************
// Copyright (c) 2008 Australian Nuclear Science and Technology Organisation.
// All rights reserved. This program and the accompanying materials
// are made available under the terms of the Eclipse Public License v1.0 
// which accompanies this distribution, and is available at
// http://www.eclipse.org/legal/epl-v10.html
// 
// Contributors: 
//    Norman Xiong (nxi@Bragg Institute) - initial API and implementation
//    ClÃ©ment Rodriguez (clement.rodriguez@synchrotron-soleil.fr) - API evolution
// ****************************************************************************
package org.cdma.interfaces;

import org.cdma.exception.ShapeNotMatchException;
import org.cdma.internal.IModelObject;

/**
 * @brief The IDimension interface is used to describe values of an axis of a data item's array.
 */ 

/// @cond pluginAPI

/**
 * @note When developing a plug-in consider using the data format engine's implementation. 
 * You should <b>redefine this interface implementation</b>, only in case of <b>very specific needs.</b>
 * <p>
 */

/// @endcond pluginAPI

/**
 * A dimension is used to define the array axis values for a data item. The shape of an array
 * can be due, for example, to a motor movement the dimension will describe its position along that axis.
 * It may be shared among data items, which provides a simple yet powerful way of associating
 * data items.
 * 
 * @author nxi
 * 
 */
public interface IDimension extends IModelObject {

    /**
     * Returns the name of this Dimension; may be null. A Dimension with a null
     * name is called "anonymous" and must be private. Dimension names are
     * unique within a Group.
     * 
     * @return String object
     */
    String getName();

    /**
     * Get the length of the Dimension.
     * 
     * @return integer value
     */
    int getLength();

    /**
     * If unlimited, then the length can increase; otherwise it is immutable.
     * 
     * @return true or false
     */
    boolean isUnlimited();

    /**
     * If variable length, then the length is unknown until the data is read.
     * 
     * @return true or false
     */
    boolean isVariableLength();

    /**
     * If this Dimension is shared, or is private to a Variable. All Dimensions
     * in NetcdfFile.getDimensions() or Group.getDimensions() are shared.
     * Dimensions in the Variable.getDimensions() may be shared or private.
     * 
     * @return true or false
     */
    boolean isShared();

    /**
     * Get the coordinate variables or coordinate variable aliases if the
     * dimension has any, else return an empty list. A coordinate variable has
     * this as its single dimension, and names this Dimensions's the
     * coordinates. A coordinate variable alias is the same as a coordinate
     * variable, but its name must match the dimension name. If numeric,
     * coordinate axis must be strictly monotonically increasing or decreasing.
     * 
     * @return IArray containing coordinates
     */
    IArray getCoordinateVariable();

    /**
     * Instances which have same contents are equal.
     * 
     * @param oo Object
     * @return true or false
     */
    boolean equals(Object oo);

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
     * Dimensions with the same name are equal.
     * 
     * @param o compare to this Dimension
     * @return 0, 1, or -1
     */
    int compareTo(Object o);

    /**
     * String representation.
     * 
     * @param strict boolean type
     * @return String object
     */
    @Deprecated
    String writeCDL(boolean strict);

    /**
     * Set whether this is unlimited, meaning length can increase.
     * 
     * @param b boolean type
     */
    void setUnlimited(boolean b);

    /**
     * Set whether the length is variable.
     * 
     * @param b boolean type
     */
    void setVariableLength(boolean b);

    /**
     * Set whether this is shared.
     * 
     * @param b boolean type
     */
    void setShared(boolean b);

    /**
     * Set the Dimension length.
     * 
     * @param n integer value
     */
    void setLength(int n);

    /**
     * Rename the dimension.
     * 
     * @param name String object
     */
    void setName(String name);

    /**
     * Set coordinates values for this dimension.
     * 
     * @param array with new coordinates
     */
    void setCoordinateVariable(IArray array) throws ShapeNotMatchException;

}
