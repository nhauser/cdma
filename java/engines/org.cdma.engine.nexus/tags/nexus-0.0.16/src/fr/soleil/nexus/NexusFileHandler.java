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

import org.nexusformat.NexusException;
import org.nexusformat.NexusFile;

public class NexusFileHandler extends NexusFile {
    public NexusFileHandler(String filename, int access) throws NexusException {
        super(filename, access);
    }

    /**
     * getSubItemName Returns the name of the item having the given class name
     * 
     * @param iIndex index of the sub-item to open (class name dependent)
     * @param sNodeClass class name of the sub-item to open
     * @return item's name
     * @throws NexusException
     * @note the first item has index number 0
     */
    public String getSubItemName(int iIndex, String sNodeClass) throws NexusException {
        if (handle < 0)
            throw new NexusException("NAPI-ERROR: File not open");
        String names[] = new String[2];
        int i = 0;
        while (nextentry(handle, names) != -1) {
            if (names[1].equals(sNodeClass)) {
                if (i == iIndex) {
                    return names[0];
                }
                i++;
            }
        }
        return null;
    }

    /**
     * Will make the loading of the NeXusAPI without crashing the whole system.
     * Send an error if the NeXus API can't be found physiqcally
     * Send an exception if the API has'nt been installed
     */
    static public void loadAPI() {
        // do nothing
    }
}
