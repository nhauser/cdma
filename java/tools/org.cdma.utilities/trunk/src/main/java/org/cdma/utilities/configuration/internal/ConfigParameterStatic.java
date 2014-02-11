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
package org.cdma.utilities.configuration.internal;

import org.cdma.interfaces.IDataset;

/**
 * <b>ConfigParameterStatic implements ConfigParameter</b><br>
 * 
 * The aim of that class is to set parameters that are independent of the content
 * of the IDataset. It means that those parameters are only specific to the data
 * model.
 * <p>
 * <b>The parameter is a constant</b> (CriterionType.CONSTANT)<br/>
 * This class is used by ConfigDataset when the plug-in asks for a specific value for that data model. For example is
 * the debug mode activate or a size of cache... ?<br>
 * It corresponds to "set" DOM element in "java" part of the "plugin" section in the plugin configuration file.
 * 
 * @see ConfigParameter
 * @see CriterionType
 * 
 * @author rodriguez
 */
public final class ConfigParameterStatic implements ConfigParameter {
    private final String mName; // Parameter name
    private final String mValue; // Parameter value

    public ConfigParameterStatic(String name, String value) {
        mName = name;
        mValue = value;

    }

    @Override
    public String getValue(IDataset dataset) {
        return mValue;
    }

    @Override
    public String getName() {
        return mName;
    }

    @Override
    public CriterionType getType() {
        return CriterionType.CONSTANT;
    }

    @Override
    public String toString() {
        return "Name: " + mName + " Type: " + CriterionType.CONSTANT + " Value: " + mValue;
    }
}
