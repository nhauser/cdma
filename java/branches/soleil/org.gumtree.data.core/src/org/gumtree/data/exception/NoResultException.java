package org.gumtree.data.exception;

public class NoResultException extends Exception {

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
