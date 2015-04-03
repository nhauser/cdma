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
        boolean result = item.hasAttribute(mName, mValue);
        return result;
    }

    @Override
    public String toString() {
        return "FilterAttribute: " + mName + " = " + mValue;
    }

    @Override
    public boolean equals(Object filter) {
        boolean result = false;

        if( filter instanceof FilterAttribute ) {
            if( mName == null ) {
                result =  ( ((FilterAttribute) filter).mName == null );
            }
            else {
                result = ( mName.equals( ((FilterAttribute) filter).mName ) );
            }

            if( result ) {
                if( mValue == null ) {
                    result =  ( ((FilterAttribute) filter).mValue == null );
                }
                else {
                    result = ( mValue.equals( ((FilterAttribute) filter).mValue ) );
                }
            }
        }
        return result;
    }

    public String getName() {
        return this.mName;
    }

    public String getValue() {
        return this.mValue;
    }

}
