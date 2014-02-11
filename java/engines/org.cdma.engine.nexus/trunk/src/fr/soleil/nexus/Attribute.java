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

public class Attribute {

 	/**
 	 * length is the length of the attribute.
 	 */
	public int length;
	
	/**
	 * type is the number type of the attribute.
     */
	public int type;
	
	/**
	 * name of the attribute
	 */
	public String name;
	
	/**
	 * value of the attribute
	 */
	protected Object value;
	
	protected Attribute(String name, int length, int type, Object value) {
		this.name = name;
		this.length = length;
		this.type = type;
		this.value = value;
	}
}
