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
package org.cdma.plugin.soleil.edf.utils;

public class StringUtils {

    public static boolean isSameString(String str1, String str2) {
        if (str1 == null) {
            return (str2 == null);
        }
        else {
            return str1.equals(str2);
        }
    }

    public static boolean isSameStringIgnoreCase(String str1, String str2) {
        if (str1 == null) {
            return (str2 == null);
        }
        else {
            return str1.equalsIgnoreCase(str2);
        }
    }

}
