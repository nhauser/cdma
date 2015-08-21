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
package org.cdma.utilities.performance.internal;

import java.util.Date;

import org.cdma.utilities.performance.PostTreatment;


public class DelayedTreatment {
	private PostTreatment process;
	private long delay;
	
	public DelayedTreatment(PostTreatment process, long delay) {
		this.process = process;
		this.delay = delay;
	}
	
	public PostTreatment getTreatment() {
		return process;
	}
	
	public long getDelay() {
		return delay;
	}
	
	public String toString() {
		return "DelayedTreatment( "+ process.toString() +", " + new Date(delay).toString() + "  )"; 
	}
}
