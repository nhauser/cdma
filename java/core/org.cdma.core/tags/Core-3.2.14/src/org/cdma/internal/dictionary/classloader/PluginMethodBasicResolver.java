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
// Copyright (c) 2008 Australian Nuclear Science and Technology Organisation.
// All rights reserved. This program and the accompanying materials
// are made available under the terms of the Eclipse Public License v1.0 
// which accompanies this distribution, and is available at
// http://www.eclipse.org/legal/epl-v10.html
// 
// Contributors: 
//    Clement Rodriguez (clement.rodriguez@synchrotron-soleil.fr) - initial API and implementation
// ****************************************************************************
package org.cdma.internal.dictionary.classloader;

/// @cond internal
/**
* @brief The PluginMethodBasicResolver implements IPluginMethodResolver interface.
* 
* The PluginMethodBasicResolver is used to discover plug-in methods that will be 
* used by the CDMA it is based on the Java ServiceLoader. 
*
*/

import java.util.ServiceLoader;
import org.cdma.dictionary.IPluginMethod;

public class PluginMethodBasicResolver implements IPluginMethodResolver {
   
    public void discoverPluginMethods(PluginMethodManager manager) {
        ServiceLoader<IPluginMethod> methods = ServiceLoader.load(IPluginMethod.class);
        for (IPluginMethod pluginMethod : methods) {
            manager.registerPluginMethod(pluginMethod);
        }
    }
    
}

/// @endcond internal
