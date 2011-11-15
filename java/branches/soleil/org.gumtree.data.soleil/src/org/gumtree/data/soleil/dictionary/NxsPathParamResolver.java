package org.gumtree.data.soleil.dictionary;

import java.util.regex.Pattern;

import org.gumtree.data.Factory;
import org.gumtree.data.IFactory;
import org.gumtree.data.dictionary.IPathParamResolver;
import org.gumtree.data.dictionary.IPathParameter;
import org.gumtree.data.dictionary.impl.Path;
import org.gumtree.data.dictionary.impl.PathParameter;
import org.gumtree.data.interfaces.IContainer;
import org.gumtree.data.soleil.NxsFactory;
import org.gumtree.data.utils.Utilities.ParameterType;

public class NxsPathParamResolver implements IPathParamResolver {
	private Path   m_path;
	private String m_pathSeparator;
	
	public NxsPathParamResolver(IFactory factory, Path path) {
		m_pathSeparator = factory.getPathSeparator();
		m_path = path;
	}
	

	@Override
	public IPathParameter resolvePathParameter(IContainer item) {
		IPathParameter result = null;
		String[] groupParts = item.getName().split( Pattern.quote(m_pathSeparator) );
		String[] pathParts  = m_path.getValue().split( Pattern.quote(m_pathSeparator) );
		String part, param, buff;
		
		// Parse the path
		for( int depth = 0; depth < groupParts.length; depth++ ) {
			if( depth < pathParts.length ) {
				part  = groupParts[depth];
				param = pathParts[depth];
				
				// If a parameter is defined
				if( param.matches( ".*(" + m_path.PARAM_PATTERN  + ")+.*") ) {
					// Try to determine the parameter result
					buff = param.replaceFirst("^(.*)(" + m_path.PARAM_PATTERN + ".*)$", "$1");
					part = part.substring(buff.length());
					buff = param.replaceAll(".*" + m_path.PARAM_PATTERN, "");
					part = part.replaceFirst(Pattern.quote(buff), "");
					buff = param.replaceAll(".*" + m_path.PARAM_PATTERN + ".*", "$1");
					result = new PathParameter(Factory.getFactory(getFactoryName()), ParameterType.SUBSTITUTION, buff, part);
					break;
				}
			}
		}
		return result;
	}

	@Override
	public String getFactoryName() {
		return NxsFactory.NAME;
	}

}
