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
package org.cdma.plugin.soleil.nexus.internal;

import java.io.File;

import org.cdma.dictionary.LogicalGroup;
import org.cdma.engine.hdf.navigation.HdfDataset;
import org.cdma.plugin.soleil.nexus.NxsFactory;

public class NexusDatasetImpl extends HdfDataset {

    // ---------------------------------------------------------
    // Internal class that concretes the abstract NexusDataset
    // ---------------------------------------------------------
    public NexusDatasetImpl(final File nexusFile, final boolean appendToExisting) throws Exception {
        super(NxsFactory.NAME, nexusFile, appendToExisting);
    }

    @Override
    public LogicalGroup getLogicalRoot() {
        return new LogicalGroup(null, null, this, false);
    }
}
