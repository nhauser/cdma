// ****************************************************************************
// Copyright (c) 2008 Australian Nuclear Science and Technology Organisation.
// All rights reserved. This program and the accompanying materials
// are made available under the terms of the Eclipse Public License v1.0 
// which accompanies this distribution, and is available at
// http://www.eclipse.org/legal/epl-v10.html
// 
// Contributors: 
//    Clement Rodriguez (clement.rodriguez@synchrotron-soleil.fr) - initial API and implementation
// ****************************************************************************
package org.cdma.internal.dictionary;

/// @cond internal
/**
 * @brief The PluginMethodManager stores and retrieve all IPluginMethod implementation for all plug-in.
 * 
 * The PluginMethodManager class is the central point to access IPluginMethod. 
 * It is a singleton pattern. It knows each IPluginMethod for each plug-ins, once
 * instantiated it won't search anymore for new ones.
 */

import java.util.Map;
import java.util.TreeMap;

import org.cdma.dictionary.IPluginMethod;

public class PluginMethodManager {
    private static final String CLASS_OSGI_PLUGIN_METHOD_RESOLVER = "org.cdma.internal.PluginMethodOsgiResolver";
    private static final String CLASS_OSGI_BUNDLE_CONTEXT = "org.osgi.framework.BundleContext";
    private static PluginMethodManager manager; // Singleton pattern
    
    // Map of map: Map< "plug-in name" , Map< "method name", IPluginMethod> >
    private Map<String, Map<String, IPluginMethod> > methodRegistry;

    public static PluginMethodManager instantiate() {
        synchronized (PluginMethodManager.class ) {
            if (manager == null) {
                manager = new PluginMethodManager();
            }
        }
        return manager;
    }
    
    /**
     * Return the IPluginMethod named pluginMethod for the given plug-in name.
     * 
     * @param factoryName name of the plug-in in which to seek the plugin method
     * @param pluginMethod name of the IPluginMethod implementation to call
     * 
     * @return IPluginMethod implementation or null if not found
     */
    public IPluginMethod getPluginMethod(String factoryName, String pluginMethod) {
        return getPluginMethodMap(factoryName).get(pluginMethod);
    }

    /**
     * Register a IPluginMethod implementation so it can be called later
     * 
     * @param pluginMethod
     * 
     */
    public void registerPluginMethod(IPluginMethod pluginMethod) {
        Map< String, IPluginMethod > methodMap = getPluginMethodMap( pluginMethod.getFactoryName() );
        methodMap.put(pluginMethod.getClass().getSimpleName(), pluginMethod);
    }

    
    // ---------------------------------------------------------
    /// Private methods
    // ---------------------------------------------------------
    private Map<String, IPluginMethod> getPluginMethodMap(String factory) {
        Map< String, IPluginMethod > methodMap = methodRegistry.get(factory);
        
        if( methodMap == null ) {
            methodMap = new TreeMap<String, IPluginMethod>();
            methodRegistry.put(factory, methodMap);
        }
        return methodMap;
    }

    private void discoverPluginMethods() {
        // Use basic factory resolver
        PluginMethodBasicResolver basicResolver = new PluginMethodBasicResolver();
        basicResolver.discoverPluginMethods(this);

        // Use osgi factory resolver if available
        try {
            // Check to see if OSGi classes are
            // available before loading the factory
            Class<?> osgiClass = Class.forName(CLASS_OSGI_BUNDLE_CONTEXT);
            if (osgiClass != null) {
                // Use reflection in case OSGi is not available at runtime
                IPluginMethodResolver osgiResolver = (IPluginMethodResolver) Class
                        .forName(CLASS_OSGI_PLUGIN_METHOD_RESOLVER).newInstance();
                osgiResolver.discoverPluginMethods(this);
            }
        } catch (Exception e) {
            // Don't worry if we can't find the osgi resolver
        }
    }
    
    private PluginMethodManager() {
        methodRegistry = new TreeMap<String, Map<String, IPluginMethod> >();
        discoverPluginMethods();
    }

}

/// @endcond internal