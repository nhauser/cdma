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
package org.cdma.plugin.soleil.internal;

import java.io.File;

import org.cdma.dictionary.LogicalGroup;
import org.cdma.engine.nexus.navigation.NexusDataset;
import org.cdma.exception.FileAccessException;
import org.cdma.plugin.soleil.NxsFactory;

public class NexusDatasetImpl extends NexusDataset {

    // ---------------------------------------------------------
    // Internal class that concretes the abstract NexusDataset
    // ---------------------------------------------------------
    public NexusDatasetImpl(File nexusFile, boolean resetBuffer) throws FileAccessException {
        super(NxsFactory.NAME, nexusFile, resetBuffer);
    }

    @Override
    public LogicalGroup getLogicalRoot() {
        return new LogicalGroup(null, null, this, false);
    }

}
