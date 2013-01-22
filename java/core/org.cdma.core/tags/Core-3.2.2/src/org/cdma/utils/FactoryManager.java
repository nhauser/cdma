// ****************************************************************************
// Copyright (c) 2008 Australian Nuclear Science and Technology Organisation.
// All rights reserved. This program and the accompanying materials
// are made available under the terms of the Eclipse Public License v1.0 
// which accompanies this distribution, and is available at
// http://www.eclipse.org/legal/epl-v10.html
// 
// Contributors: 
//    Norman Xiong (nxi@Bragg Institute) - initial API and implementation
// ****************************************************************************
package org.cdma.utils;

/// @cond internal

import java.util.Collections;
import java.util.Map;
import java.util.TreeMap;

import org.cdma.IFactory;
import org.cdma.internal.BasicFactoryResolver;

public class FactoryManager implements IFactoryManager {

    // TODO: Make it configurable in system properties
    private static final String CLASS_OSGI_FACTORY_RESOLVER = "org.cdma.internal.OsgiFactoryResolver";

    private static final String CLASS_OSGI_BUNDLE_CONTEXT = "org.osgi.framework.BundleContext";

    // System property for default factory 
    private static final String PROP_DEFAULT_FACTORY = "gumtree.data.defaultFactory";

    private Map<String, IFactory> factoryRegistry;
    private String version;

    /**
     * 
     * @param cdmaVersion filter on API version to apply when loading plug-ins 
     */
    public FactoryManager(String cdmaVersion) {
        factoryRegistry = new TreeMap<String, IFactory>();
        version = cdmaVersion;
        discoverFactories();
    }

    public void registerFactory(String name, IFactory factory) {
    	int[] cdmaVersion = parseVersion( factory.getCDMAVersion() );
    	
    	// Check if a version is mentioned for registry
    	boolean register = true;
    	if( version != null && ! version.isEmpty() ) {
    		// Check the factory's version is compatible with the API version
    		int[] coreVersion = parseVersion( version );
    		
    		if( coreVersion[0] != cdmaVersion[0] ) {
    			register = false;
    		}
    	}

    	if( register ) {
    		factoryRegistry.put(name, factory);
    	}
    }

    public IFactory getFactory() {
        IFactory factory = null;
        String defaultFactoryName = System.getProperty(PROP_DEFAULT_FACTORY);
        if (defaultFactoryName != null) {
            // If default factory is specified
            factory = factoryRegistry.get(defaultFactoryName);
        }
        if (factory == null && !factoryRegistry.isEmpty()) {
            // If default factory is not specified or doesn't exist
            factory = factoryRegistry.values().iterator().next();
        }
        return factory;
    }

    public IFactory getFactory(String name) {
        return factoryRegistry.get(name);
    }

    public Map<String, IFactory> getFactoryRegistry() {
        return Collections.unmodifiableMap(factoryRegistry);
    }

    // ---------------------------------------------------------
    // / Private methods
    // ---------------------------------------------------------
    private void discoverFactories() {
        // Use basic factory resolver
        IFactoryResolver basicResolver = new BasicFactoryResolver();
        basicResolver.discoverFactories(this);

        // Use osgi factory resolver if available
        try {
            // [ANSTO][Tony][2011-05-25] Check to see if OSGi classes are
            // available before loading the factory
            Class<?> osgiClass = Class.forName(CLASS_OSGI_BUNDLE_CONTEXT);
            if (osgiClass != null) {
                // Use reflection in case OSGi is not available at runtime
                IFactoryResolver osgiResolver = (IFactoryResolver) Class
                        .forName(CLASS_OSGI_FACTORY_RESOLVER).newInstance();
                osgiResolver.discoverFactories(this);
            }
        } catch (Exception e) {
            // Don't worry if we can't find the osgi resolver
        }
    }
    
    /**
     * Parse the given string to extract a number version
     * 
     * @param version in string format X.Y.Z
     * @return a int[] of length 3 with major version in first cell.
     */
    private int[] parseVersion(String version) {
    	int[] result = new int[3];
    	
    	String[] numbers = version.split("[^0-9]");
    	int i = 0;
    	for( String num : numbers ) {
    		if( ! num.isEmpty() ) {
	    		result[i] = Integer.parseInt(num);
	    		i++;
	    		if( i >= 3) {
	    			break;
	    		}
    		}
    	}
    	return result;
    }
}

/// @endcond internal