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
 * @brief The FilterAttribute class is used to select a specific IContainer.
 * 
 * The IFilter implementation FilterAttribute defines a filter that will 
 * test the presence of a specific attribute having the right name and 
 * value (in String). 
 * 
 * @author rodriguez
 */

public class FilterAttribute implements IFilter {
    private String mName;
    private String mValue;
    
    public FilterAttribute(String name, String value) {
        mName = name;
        mValue = value;
    }
    
    @Override
    public boolean matches(IContainer item) {
        return item.hasAttribute(mName, mValue);
    }
    
    @Override
    public String toString() {
        return "FilterAttribute: " + mName + " = " + mValue;
    }

}
