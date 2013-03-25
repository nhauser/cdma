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
 */
public class DimensionNotSupportedException extends CDMAException {

    /**
     * 
     */
    private static final long serialVersionUID = -8704918780837836462L;

    /**
     * 
     */
    public DimensionNotSupportedException() {
    }

    /**
     * @param message String value
     */
    public DimensionNotSupportedException(final String message) {
        super(message);
    }

    /**
     * @param cause Throwable object
     */
    public DimensionNotSupportedException(final Throwable cause) {
        super(cause);
    }

    /**
     * @param message String value
     * @param cause Throwable object
     */
    public DimensionNotSupportedException(final String message,
            final Throwable cause) {
        super(message, cause);
    }

}
