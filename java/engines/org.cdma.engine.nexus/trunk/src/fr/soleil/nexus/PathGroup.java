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

public class PathGroup extends PathNexus {

    /**
     * PathGroup
     * Create an object PathGroup
     * 
     * @param sGroups array containing name of each group.
     * @note group's class can be specified by adding "<" and ">" to a class name: i.e. "my_entry<NXentry>"
     * @note be aware that it's better not to force the group class by default they are mapped by the API
     */

    public PathGroup(String[] sGroups) {
        super(sGroups);
    }

    public PathGroup(PathNexus pnPath) {
        super(pnPath.getGroups());
        this.setFile(pnPath.getFilePath());
    }

    protected PathGroup(String sAcquiName) {
        super(new String[] { sAcquiName });
    }

    static public PathGroup Convert(PathNexus pnPath) {
        return new PathGroup(pnPath);
    }

}
