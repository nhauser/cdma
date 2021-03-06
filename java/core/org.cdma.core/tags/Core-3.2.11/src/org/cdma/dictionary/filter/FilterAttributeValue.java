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
 *        St�phane Poirier (SOLEIL Synchrotron) - API v2 design and conception
 * 	Clement Rodriguez (ALTEN for SOLEIL Synchrotron) - API evolution
 * 	Gregory VIGUIER (SOLEIL Synchrotron) - API evolution
 ******************************************************************************/
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

import java.util.List;

/**
 * @brief The FilterAttributeValue class is used to select a specific IContainer.
 * 
 * The IFilter implementation FilterAttributeValue defines a filter that will 
 * test the presence of any attribute having the right value (in String).
 * It doesn't care about the attribute's name. 
 * 
 * @author rodriguez
 */


import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IContainer;

public class FilterAttributeValue implements IFilter {
    private String mValue;
    
    public FilterAttributeValue(String value) {
        mValue = value;
    }
    
    @Override
    public boolean matches(IContainer item) {
        boolean result = false;
        
        List<IAttribute> list = item.getAttributeList();
        String test;
        for( IAttribute attribute : list ) {
            test = attribute.getStringValue();
            if( test != null && test.equals(mValue) ) {
                result = true;
                break;
            }
        }
        return result;
    }
    
    @Override
    public String toString() {
        return "FilterAttributeValue: " + mValue;
    }
    
    @Override
    public boolean equals(Object filter) {
        boolean result = false;
        
        if( filter instanceof FilterAttributeValue ) {
            if( mValue == null ) {
                result =  ( ((FilterAttributeValue) filter).mValue == null );
            }
            else {
                result = ( mValue.equals( ((FilterAttributeValue) filter).mValue ) );
            }
        }
        return result;
    }
}
