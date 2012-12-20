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

public class NoResultException extends CDMAException {

    private static final long serialVersionUID = 9120780040147247194L;

    public NoResultException() {
        super();
    }

    public NoResultException(String message) {
        super(message);
    }

    public NoResultException(Throwable cause) {
        super(cause);
    }

    public NoResultException(String message, Throwable cause) {
        super(message, cause);
    }

}
