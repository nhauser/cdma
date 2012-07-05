package org.gumtree.data.internal;

import java.util.ServiceLoader;

import org.gumtree.data.IFactory;
import org.gumtree.data.utils.IFactoryManager;
import org.gumtree.data.utils.IFactoryResolver;

public class BasicFactoryResolver implements IFactoryResolver {

	public void discoverFactories(IFactoryManager manager) {
		ServiceLoader<IFactory> factories = ServiceLoader.load(IFactory.class);
		for (IFactory factory : factories) {
			manager.registerFactory(factory.getName(), factory);
		}
	}

}
