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
package org.cdma.internal.dictionary.readers;

import java.util.HashMap;
import java.util.Map;

import org.cdma.internal.dictionary.solvers.ItemSolver;

public class DataMapping implements Cloneable {
	private String mVersion;
	private Map<String, ItemSolver> mSolvers;
	
	public DataMapping( String version ) {
		mVersion = version;
		mSolvers = new HashMap<String,ItemSolver>();
	}

	public void addSolver(String keyID, ItemSolver itemSolver) {
		mSolvers.put(keyID, itemSolver);
	}
	
	public DataMapping clone() throws CloneNotSupportedException {
		DataMapping clone = new DataMapping(mVersion);
		clone.mSolvers = new HashMap<String, ItemSolver>(mSolvers);
		return clone;
	}
	
	public Map<String, ItemSolver> getSolvers() {
		return mSolvers;
	}
}
