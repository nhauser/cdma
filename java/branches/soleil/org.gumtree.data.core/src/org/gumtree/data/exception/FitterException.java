/*******************************************************************************
 * Copyright (c) 2010 Australian Nuclear Science and Technology Organisation.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0 
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 * 
 * Contributors: 
 *    Norman Xiong (nxi@Bragg Institute) - initial API and implementation
 ******************************************************************************/
package org.gumtree.data.exception;

/**
 * @author nxi 
 */
public class FitterException extends Exception {

  /**
   * 
   */
  private static final long serialVersionUID = -8055965069428891892L;

  /**
   * 
   */
  public FitterException() {
  }

  /**
   * @param message String value
   */
  public FitterException(final String message) {
    super(message);
  }

  /**
   * @param cause Throwable object
   */
  public FitterException(final Throwable cause) {
    super(cause);
  }

  /**
   * @param message String value
   * @param cause Throwable object
   */
  public FitterException(final String message, final Throwable cause) {
    super(message, cause);
  }

}
