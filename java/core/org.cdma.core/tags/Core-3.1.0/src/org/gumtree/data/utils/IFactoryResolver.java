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
package org.gumtree.data.utils;

/// @cond internal

/**
 * @brief The IFactoryResolver is used to discover factories that will be used by the CDMA.
 * 
 * The factory resolver aims to find all factories.
 *
 */

public interface IFactoryResolver {

    public void discoverFactories(IFactoryManager manager);

}

/// @endcond internal