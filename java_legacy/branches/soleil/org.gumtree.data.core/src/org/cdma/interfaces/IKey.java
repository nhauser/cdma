// ****************************************************************************
// Copyright (c) 2008 Australian Nuclear Science and Technology Organisation.
// All rights reserved. This program and the accompanying materials
// are made available under the terms of the Eclipse Public License v1.0 
// which accompanies this distribution, and is available at
// http://www.eclipse.org/legal/epl-v10.html
// 
// Contributors: 
//    Norman Xiong (nxi@Bragg Institute) - initial API and implementation
//    Clément Rodriguez (clement.rodriguez@synchrotron-soleil.fr) - API evolution
// ****************************************************************************
package org.cdma.interfaces;

import java.util.List;

import org.cdma.dictionary.filter.IFilter;
import org.cdma.internal.IModelObject;

/**
 * @brief The IKey is used by group to interrogate the dictionary.
 */

/// @cond pluginAPI

/**
 * @note When developing a plug-in consider using the engine's implementation.
 * You should <b>redefine this interface implementation</b>, only in case of <b>very specific needs.</b>
 * <p>
 */

/// @endcond pluginAPI

/// @cond engineAPI

/**
 * @note When developing a data consider using the default implementation in the dictionary.impl package.
 * You should <b>redefine this interface implementation</b>, only in case of <b>very specific needs.</b>
 * <p>
 */

/// @endcond engineAPI

/**
 * The key's name corresponds to an entry in the dictionary. This entry 
 * targets a path in the currently explored document. The group will open it.
 * <p>
 * The IKey can carry some filters to help group to decide which node is relevant.
 * The filters can specify an order index to open a particular type of node, an 
 * attribute, a part of the name... 
 * <p>
 * @author rodriguez
 */

public interface IKey extends IModelObject, Cloneable, Comparable<Object> {
    /**
     * Get the entry name in the dictionary that will be 
     * searched when using this key. 
     * 
     * @return the name of this key
     */
    String getName();

    /**
     * Set the entry name in the dictionary that will be 
     * searched when using this key. 
     * 
     * @param name of this key
     */
    void setName(String name);

    /**
     * Return true if both key have similar names. Filters are not compared. 
     * 
     * @param key to compare
     * @return true if both keys have same name
     */
    boolean equals(Object key);

    /**
     * Get the list of parameters that will be applied when using this key.
     * 
     * @return list of IFilter
     * @see org.cdma.dictionary.filter.IFilter
     */
    List<IFilter> getFilterList();

    /**
     * Add a IFilter to this IKey that will be used when 
     * searching an object with this key.
     * 
     * @param filter to be applied
     * @note work as a FILO
     * @see org.cdma.dictionary.filter.IFilter
     */
    void pushFilter(IFilter filter);
    
    /**
     * Remove a IFilter from this IKey that will be used when 
     * searching an object.
     * 
     * @return filter that won't be applied anymore
     * @note work as a FILO
     * @see org.cdma.dictionary.filter.IFilter
     */
    IFilter popFilter();

    String toString();

    /**
     * Copy entirely the key : name and filters are cloned
     * @return a copy of this key
     */
    IKey clone();
}
