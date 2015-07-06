/*******************************************************************************
 * Copyright (c) 2008 - ANSTO/Synchrotron SOLEIL
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 * 
 * Contributors:
 * Norman Xiong (nxi@Bragg Institute) - initial API and implementation
 * Tony Lam (nxi@Bragg Institute) - initial API and implementation
 * Majid Ounsy (SOLEIL Synchrotron) - API v2 design and conception
 * Stéphane Poirier (SOLEIL Synchrotron) - API v2 design and conception
 * Clement Rodriguez (ALTEN for SOLEIL Synchrotron) - API evolution
 * Gregory VIGUIER (SOLEIL Synchrotron) - API evolution
 ******************************************************************************/
// ****************************************************************************
// Copyright (c) 2008 Australian Nuclear Science and Technology Organisation.
// All rights reserved. This program and the accompanying materials
// are made available under the terms of the Eclipse Public License v1.0
// which accompanies this distribution, and is available at
// http://www.eclipse.org/legal/epl-v10.html
//
// Authors:
//     - Gregouze The Tarlouze at synchrotron-soleil.fr
//     - Rodriguez The Merguez at synchrotron-soleil.fr
// ****************************************************************************
package org.cdma.interfaces;

/**
 * 
 */
public interface INode {

    /**
     * 
     */
    public String getNodeName();

    /**
     * 
     */
    public String getName();

    /**
     * 
     */
    public boolean matchesNode(INode node);

    /**
     * 
     */
    public boolean matchesPartNode(INode node);

    /**
     * 
     */
    public boolean isGroup();
}
