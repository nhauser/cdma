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

import java.util.ServiceLoader;

import org.gumtree.data.IFactory;
import org.gumtree.data.utils.IFactoryManager;
import org.gumtree.data.utils.IFactoryResolver;

public class BasicFactoryResolver implements IFactoryResolver {

    public void discoverFactories(IFactoryManager manager) {
        ServiceLoader<IFactory> factories = ServiceLoader.load(IFactory.class);
        for (IFactory factory : factories) {
            manager.registerFactory(factory.getName(), factory);
        }
    }

}
