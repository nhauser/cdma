// ****************************************************************************
// Copyright (c) 2010 Australian Nuclear Science and Technology Organisation.
// All rights reserved. This program and the accompanying materials
// are made available under the terms of the Eclipse Public License v1.0 
// which accompanies this distribution, and is available at
// http://www.eclipse.org/legal/epl-v10.html
// 
// Contributors
//    Clement Rodriguez - initial API and implementation
// ****************************************************************************
package org.cdma.dictionary.filter;

import org.cdma.interfaces.IContainer;

/**
 * @brief The IFilter interface is used to make a selective choice when browsing a IDataset.
 * 
 * A IFilter represents conditions that permits identifying a specific node using the 
 * <i>Extended Dictionary Mechanism</i>. 
 * When a given IKey can return several IContainer, the IFilter applied on that key
 * will make possible to define which one is relevant.
 * <p>
 * The filter implementation only defines a method that will test the IContainer.
 * If the IContainer is matching the filter then it will be returned.
 * <p>
 * The filter can consist in the existence of an attribute or whatever that should be relevant to 
 * formally identify a specific node while several are possible according to the path.
 * 
 * @author rodriguez
 */

public interface IFilter {
    /**
     * Verify if the given container matches this filter.
     * 
     * @param item to test properties
     * @return true if the IContainer has the properties the filter expects
     */
    boolean matches(IContainer item);
    
    /**
     * Equality test of to filters implementation.
     * 
     * @param filter to be compared to this one
     * @return true if both IFilter implementation have same type and filtering properties 
     */
    public boolean equals(Object filter);
    
}
