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
