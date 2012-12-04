// +======================================================================
// $Source: /cvsroot/tango-cs/tango/tools/mambo/tools/xmlhelpers/XMLUtils.java,v
// $
//
// Project: Tango Archiving Service
//
// Description: Java source code for the class XMLUtils.
// (Claisse Laurent) - 5 juil. 2005
//
// $Author: pierrejoseph $
//
// $Revision: 1.1 $
//
// $Log: XMLUtils.java,v $
// Revision 1.1 2007/02/01 14:07:17 pierrejoseph
// getAttributesToDedicatedArchiver
//
// Revision 1.5 2006/11/22 10:44:14 ounsy
// corrected a NullPointer bug in loadAttributes()
//
// Revision 1.4 2006/09/22 14:52:23 ounsy
// minor changes
//
// Revision 1.3 2006/05/19 15:05:29 ounsy
// minor changes
//
// Revision 1.2 2005/11/29 18:28:26 chinkumo
// no message
//
// Revision 1.1.2.3 2005/09/26 07:52:25 chinkumo
// Miscellaneous changes...
//
// Revision 1.1.2.2 2005/09/14 15:41:44 chinkumo
// Second commit !
//
//
// copyleft : Synchrotron SOLEIL
// L'Orme des Merisiers
// Saint-Aubin - BP 48
// 91192 GIF-sur-YVETTE CEDEX
//
// -======================================================================
package org.cdma.plugin.xml;

import java.io.File;
import java.util.Hashtable;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;

import org.w3c.dom.Document;
import org.w3c.dom.NamedNodeMap;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

public class XmlUtils {


	/**
	 * @param file
	 * @return
	 * @throws Exception
	 *             8 juil. 2005
	 */
	public static Node getRootNode(File file) throws Exception {
		try {
			DocumentBuilderFactory factory = DocumentBuilderFactory
					.newInstance();

			// should we add dtd validating?
			factory.setValidating(false);
			factory.setIgnoringComments(true);

			DocumentBuilder builder = factory.newDocumentBuilder();

			// builder.setErrorHandler ( error );
			Document doc = builder.parse(file);

			return doc.getDocumentElement();
		} catch (Exception e) {
			throw e;
			// throw e;
		}
	}

	/**
	 * @param noeudATester
	 * @return 8 juil. 2005
	 */
	public static boolean isAFakeNode(Node noeudATester) {
		int typeNode = noeudATester.getNodeType();

		return (typeNode != Node.ELEMENT_NODE);

	}

	/**
	 * @param nodeToTest
	 * @return 8 juil. 2005
	 */
	public static boolean hasRealChildNodes(Node nodeToTest) {
		boolean ret = false;

		NodeList potentialChilds = nodeToTest.getChildNodes();
		for (int i = 0; i < potentialChilds.getLength(); i++) {
			Node nextNode = potentialChilds.item(i);
			if (!isAFakeNode(nextNode)) {
				return true;
			}
		}

		return ret;
	}

	/**
	 * @param noeudDOM
	 * @return
	 * @throws Exception
	 *             8 juil. 2005
	 */
	public static Hashtable<String, String> loadAttributes(Node DOMnode)
			throws Exception {
		Hashtable<String, String> retour = new Hashtable<String, String>();
		if (DOMnode.hasAttributes()) {
			NamedNodeMap listAtts = DOMnode.getAttributes();

			for (int i = 0; i < listAtts.getLength(); i++) {
				String nomAtt = listAtts.item(i).getNodeName().trim();
				String valueAtt = listAtts.item(i).getNodeValue().trim();
				retour.put(nomAtt, valueAtt);
			}
		}

		return retour;
	}

}
