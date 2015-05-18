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
// ****************************************************************************
// Copyright (c) 2008 Australian Nuclear Science and Technology Organisation.
// All rights reserved. This program and the accompanying materials
// are made available under the terms of the Eclipse Public License v1.0 
// which accompanies this distribution, and is available at
// http://www.eclipse.org/legal/epl-v10.html
// 
// Contributors: 
//    Norman Xiong (nxi@Bragg Institute) - initial API and implementation
// ****************************************************************************
package org.cdma.exception;

/**
 * @author nxi
 * 
 */
public class FileAccessException extends CDMAException {

    /**
     * 
     */
    private static final long serialVersionUID = 5360865109579178570L;

    /**
     * 
     */
    public FileAccessException() {
    }

    /**
     * @param arg0 String value
     */
    public FileAccessException(final String arg0) {
        super(arg0);
    }

    /**
     * @param arg0 Throwable object
     */
    public FileAccessException(final Throwable arg0) {
        super(arg0);
    }

    /**
     * @param arg0 String value
     * @param arg1 Throwable object
     */
    public FileAccessException(final String arg0, final Throwable arg1) {
        super(arg0, arg1);
    }

}
