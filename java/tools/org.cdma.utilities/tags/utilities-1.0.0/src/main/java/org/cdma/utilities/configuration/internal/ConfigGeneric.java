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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.cdma.interfaces.IDataset;
import org.cdma.utilities.configuration.internal.ConfigCriteria;
import org.cdma.utilities.configuration.internal.ConfigParameter;
import org.cdma.utilities.configuration.internal.ConfigParameterDynamic;
import org.cdma.utilities.configuration.internal.ConfigParameterStatic;
import org.jdom2.Element;

/**
 * ConfigDataset defines some criteria that will permit to verify the given IDataset
 * matches it. A IDataset of the plug-in has one and only one configuration.
 * <p>
 * It permits to determines some parameters that can be statically or dynamically (according
 * to some conditions or values in dataset) fixed for a that specific data model.
 * Those parameters (ConfigParameter) are resolved according the given IDataset.
 * <p>
 * Each IDataset should match a specific ConfigDataset.
 * 
 * @see ConfigParameter ConfigParameter: interface a parameter must implement
 * @see ConfigCriteria ConfigCriteria: criteria that an IDataset must respect to match a ConfigDataset
 * @author rodriguez
 *
 */

public class ConfigGeneric {
    private ConfigCriteria mCriteria;             // Criteria dataset should match for this configuration
    private Map<String, ConfigParameter> mParams; // Param name/value to set in the plug-in when using this configuration
    private String mConfigLabel;                  // Name of the configuration

    /**
     * Constructor of the dataset configuration
     * need a dom element named: "config_dataset"
     * @param dataset_model DOM element "dataset_model"
     * @param params some default parameters that can be override by this Config
     */
    public ConfigGeneric(Element dataset_model, List<ConfigParameter> params ) {
        mConfigLabel = dataset_model.getAttributeValue("name");
        mParams = new HashMap<String, ConfigParameter>();
        for( ConfigParameter param : params ) {
            mParams.put(param.getName(), param);
        }
        init(dataset_model);
    }

    /**
     * Return the label of that ConfigDataset
     * @return
     */
    public String getLabel() {
        return mConfigLabel;
    }

    /**
     * Returns the list of all parameters that can be asked for that configuration
     * @return list of ConfigParameter
     */
    public List<ConfigParameter> getParameters() {
        List<ConfigParameter> result = new ArrayList<ConfigParameter>();

        // Fills the output list with existing parameters
        for( Entry<String, ConfigParameter> entry : mParams.entrySet()) {
            result.add( entry.getValue() );
        }
        return result;
    }

    /**
     * Returns the value of the named <b>ConfigParameter</b> for the given IDataset. 
     * @param label of the parameter 
     * @param dataset used to resolve that parameter
     * @return the string value of the parameter
     */
    public String getParameter(String label, IDataset dataset) {
        String result = "";
        if( mParams.containsKey(label) ) {
            result = mParams.get(label).getValue(dataset);
        }
        return result;
    }

    /**
     * Returns the criteria a IDataset must respect to match that configuration.
     * @return ConfigCriteria object
     */
    public ConfigCriteria getCriteria() {
        return mCriteria;
    }

    /**
     * Add a parameter to that configuration
     * @param param implementing ConfigParameter interface
     */
    public void addParameter(ConfigParameter param) {
        mParams.put(param.getName(), param);
    }

    @Override
    public String toString() {
        String result = "Configuration: " + mConfigLabel + "\n";
        
        result += mCriteria + "\n";
        result += "Parameters: ";
        for( Entry<String, ConfigParameter> entry : mParams.entrySet() ) {
            result += "\n" + entry.getValue();
        }
        
        return result;
    }
    
    // ---------------------------------------------------------
    /// Private methods
    // ---------------------------------------------------------
    /**
     * Parse the DOM element "dataset_model" to initialize this object
     * @param config_dataset dom element markup "dataset_model"
     */
    private void init(Element config_dataset) {
        List<?> nodes;
        Element elem;
        Element section;
        ConfigParameter parameter;
        String name;
        String value;

        // Managing criteria
        mCriteria = new ConfigCriteria();
        section = config_dataset.getChild("criteria");
        if( section != null ) {
            mCriteria.add( section );
        }

        // Managing plug-in parameters (static ones)
        section = config_dataset.getChild("plugin");
        if( section != null ) {
            Element javaSection = section.getChild("java");
            if( javaSection != null ) {
                nodes = javaSection.getChildren("set");
                for( Object set : nodes ) {
                    elem  = (Element) set;
                    name  = elem.getAttributeValue("name");
                    value = elem.getAttributeValue("value");
                    parameter = new ConfigParameterStatic( name, value );
                    mParams.put( name, parameter );
                }
            }
        }

        // Managing dynamic parameters
        section = config_dataset.getChild("parameters");
        if( section != null ) {
            nodes = section.getChildren("parameter");
            String type;
            for( Object node : nodes ) {
                elem = (Element) node;
                name = elem.getAttributeValue("name");
                type = elem.getAttributeValue("type");

                // Dynamic parameter
                if( ! type.equals("constant") ) {
                    parameter = new ConfigParameterDynamic(elem);
                }
                // Static parameter (constant)
                else {
                    value = elem.getAttributeValue("constant");
                    parameter = new ConfigParameterStatic(name, value);
                }
                mParams.put(name, parameter);
            }
        }
    }
}
