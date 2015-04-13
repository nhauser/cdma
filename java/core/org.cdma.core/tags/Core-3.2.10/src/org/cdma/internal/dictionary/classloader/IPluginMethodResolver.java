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
* @brief The IPluginMethodResolver is used to discover plug-in methods that will be used by the CDMA.
* 
* The plug-in method resolver aims to find all IPluginMethod implementation in class path, 
* the search process is done only once at the loading of a plug-in.
*
*/

public interface IPluginMethodResolver {
    
    void discoverPluginMethods(PluginMethodManager manager);
    
}

/// @endcond internal
