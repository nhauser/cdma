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
 * @brief The FilterAttributeName class is used to select a specific IContainer.
 * 
 * The IFilter implementation FilterAttributeName defines a filter that will 
 * test the presence of an attribute having the given name. It doesn't care
 * about its value.
 * 
 * @author rodriguez
 */

public class FilterAttributeName implements IFilter {
    private String mName;
    
    public FilterAttributeName(String name) {
        mName = name;
    }
    
    @Override
    public boolean matches(IContainer item) {
        boolean result = false;
        
        if( item.getAttribute(mName) != null ) {
            result = true;
        }
        return result;
    }
    
    @Override
    public String toString() {
        return "FilterAttributeValue: " + mName;
    }
    
    @Override
    public boolean equals(Object filter) {
        boolean result = false;
        
        if( filter instanceof FilterAttributeName ) {
            if( mName == null ) {
                result =  ( ((FilterAttributeName) filter).mName == null );
            }
            else {
                result = ( mName.equals( ((FilterAttributeName) filter).mName ) );
            }
        }
        return result;
    }
}
