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
package org.gumtree.data.exception;

public class TooManyResultException extends Exception {

    private static final long serialVersionUID = 2757561816423327473L;

    public TooManyResultException() {
        super();
    }

    public TooManyResultException(String message) {
        super(message);
    }

    public TooManyResultException(Throwable cause) {
        super(cause);
    }

    public TooManyResultException(String message, Throwable cause) {
        super(message, cause);
    }

}
