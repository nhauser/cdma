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
* @brief The PluginMethodOsgiResolver implements IPluginMethodResolver interface.
* 
* The PluginMethodOsgiResolver is used to discover plug-in methods that will be 
* used by the CDMA it is based on the OSGI Bundle Service Loading. 
*
*/

import org.cdma.dictionary.IPluginMethod;
import org.cdma.internal.Activator;
import org.osgi.framework.BundleContext;
import org.osgi.framework.InvalidSyntaxException;
import org.osgi.framework.ServiceReference;

public class PluginMethodOsgiResolver implements IPluginMethodResolver {

    public void discoverPluginMethods(PluginMethodManager manager) {
        BundleContext context = Activator.getDefault().getContext();
        ServiceReference[] refs = null;
        try {
            refs = context.getServiceReferences(IPluginMethod.class.getName(), null);
        } catch (InvalidSyntaxException e) {
        }
        if (refs != null) {
            for (ServiceReference ref : refs) {
                IPluginMethod pluginMethod = (IPluginMethod) context.getService(ref);
                manager.registerPluginMethod(pluginMethod);
            }
        }
    }
    
}

/// @endcond internal