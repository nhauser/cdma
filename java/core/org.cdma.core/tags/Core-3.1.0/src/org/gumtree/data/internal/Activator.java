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
package org.gumtree.data.internal;

/// @cond internal

import org.osgi.framework.BundleActivator;
import org.osgi.framework.BundleContext;

public class Activator implements BundleActivator {

    private static Activator instance;

    private BundleContext context;

    public void start(BundleContext context) throws Exception {
        this.context = context;
        instance = this;
    }

    public void stop(BundleContext context) throws Exception {
        instance = null;
        this.context = null;
    }

    public BundleContext getContext() {
        return context;
    }

    public static Activator getDefault() {
        return instance;
    }

}

/// @endcond internal