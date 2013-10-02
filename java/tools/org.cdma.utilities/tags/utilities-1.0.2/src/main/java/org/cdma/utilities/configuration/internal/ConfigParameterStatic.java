//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
//    Clément Rodriguez (clement.rodriguez@synchrotron-soleil.fr)
// See AUTHORS file
//******************************************************************************
package org.cdma.utilities.configuration.internal;

import org.cdma.interfaces.IDataset;
import org.cdma.utilities.configuration.internal.ConfigParameter.CriterionType;

/**
 * <b>ConfigParameterStatic implements ConfigParameter</b><br>
 * 
 * The aim of that class is to set parameters that are independent of the content 
 * of the IDataset. It means that those parameters are only specific to the data 
 * model.<p> 
 * <b>The parameter is a constant</b> (CriterionType.CONSTANT)<br/>
 * This class is used by ConfigDataset when the plug-in asks for a specific value
 * for that data model. For example is the debug mode activate or a size of 
 * cache... ?<br>  
 * It corresponds to "set" DOM element in "java" part of the "plugin" section
 * in the plugin configuration file.
 *
 * @see ConfigParameter
 * @see CriterionType
 * 
 * @author rodriguez
 */
public final class ConfigParameterStatic implements ConfigParameter {
    private String         mName;     // Parameter name
    private String         mValue;    // Parameter value

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

