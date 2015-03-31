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
package fr.soleil.nexus;

public class PathInst extends PathAcqui {
    /**
     * PathInst
     * Create an object PathInst
     * 
     * @param sAcquiName name of the acquisition which the instrument will belong to
     * @param sInstName name of the instrument
     * @note group's class can be specified by adding "<" and ">" to a class name: i.e. "my_entry<NXentry>"
     * @note BE AWARE that it's better not to force the group's class. By default they are mapped by the API to apply
     *       Nexus format DTD
     */
    public PathInst(String sAcquiName, String sInstName) {
        super(new String[] { sAcquiName, "<NXinstrument>", sInstName });
    }
}
