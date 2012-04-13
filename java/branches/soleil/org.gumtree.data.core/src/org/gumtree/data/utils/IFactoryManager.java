package org.gumtree.data.utils;

import java.util.Map;

import org.gumtree.data.IFactory;

public interface IFactoryManager {

  public void registerFactory(String name, IFactory factory);
  
  public IFactory getFactory();
  
  public IFactory getFactory(String name);
  
  public Map<String, IFactory> getFactoryRegistry();
  
}
