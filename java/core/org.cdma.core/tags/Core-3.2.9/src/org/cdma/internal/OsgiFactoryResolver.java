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
//    Norman Xiong (nxi@Bragg Institute) - initial API and implementation
// ****************************************************************************
package org.cdma.internal;

/// @cond internal

import org.cdma.IFactory;
import org.cdma.utils.FactoryManager;
import org.cdma.utils.IFactoryResolver;
import org.osgi.framework.BundleContext;
import org.osgi.framework.InvalidSyntaxException;
import org.osgi.framework.ServiceReference;

public class OsgiFactoryResolver implements IFactoryResolver {

    public void discoverFactories(FactoryManager manager) {
        BundleContext context = Activator.getDefault().getContext();
        ServiceReference[] refs = null;
        try {
            refs = context.getServiceReferences(IFactory.class.getName(), null);
        } catch (InvalidSyntaxException e) {
        }
        if (refs != null) {
            for (ServiceReference ref : refs) {
                IFactory factory = (IFactory) context.getService(ref);
                manager.registerFactory(factory.getName(), factory);
                factory.processPostRecording();
            }
        }
    }

}

/// @endcond internal
