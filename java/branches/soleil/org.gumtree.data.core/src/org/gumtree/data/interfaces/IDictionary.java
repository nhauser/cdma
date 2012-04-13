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
package org.gumtree.data.interfaces;

import java.net.URI;
import java.util.List;

import org.gumtree.data.dictionary.IPath;
import org.gumtree.data.exception.FileAccessException;

/**
 * An interface for dictionary used in CDMA Group model to reference a path with
 * a key in String type. The target object in the path can be either a Group or
 * a DataItem.
 * 
 * @author nxi
 * 
 */
public interface IDictionary extends IModelObject, Cloneable {

  /**
   * Return all keys referenced in the dictionary.
   * 
   * @return a list of String objects
   */
  List<IKey> getAllKeys();
  
  /**
   * Get the path referenced by the key. If there are more than one paths are
   * referenced by the path, get the default one.
   * 
   * @param key key object
   * @return String object
   */
  IPath getPath(IKey key);

  /**
   * Return all paths referenced by the key.
   * 
   * @param key key object
   * @return a list of String objects
   */
  List<IPath> getAllPaths(IKey key);

  /**
   * Add an entry of key and path.
   * 
   * @param key key object
   * @param path String object
   */
  void addEntry(String key, String path);
  
  void addEntry(String key, IPath path);

  /**
   * Read dictionary entries from a file.
   * 
   * @param uri URI object
   * @throws FileAccessException
   *             I/O file access exception
   */
  void readEntries(URI uri) throws FileAccessException;

  /**
   * Read dictionary entries from a file.
   * 
   * @param path String object
   * @throws FileAccessException
   *             I/O file access exception
   */
  void readEntries(String path) throws FileAccessException;

  /**
   * Remove a path from an entry. If there is only one path associated with
   * the key, then remove the entry as well.
   * 
   * @param key key object
   * @param path String object
   */
  void removeEntry(String key, String path);

  /**
   * Remove an entry from the dictionary.
   * 
   * @param key key object
   */
  void removeEntry(String key);

  /**
   * @param key key object
   * @return true or false
   */
  boolean containsKey(String key);

  /**
   * Clone the dictionary in a new object.
   * 
   * @return new Dictionary object
   * @throws CloneNotSupportedException
   *             failed to clone
   */
  IDictionary clone() throws CloneNotSupportedException;

}
