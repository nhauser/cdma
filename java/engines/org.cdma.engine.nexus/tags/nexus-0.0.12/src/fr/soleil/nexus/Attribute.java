package fr.soleil.nexus;

public class Attribute {

 	/**
 	 * length is the length of the attribute.
 	 */
	public int length;
	
	/**
	 * type is the number type of the attribute.
     */
	public int type;
	
	/**
	 * name of the attribute
	 */
	public String name;
	
	/**
	 * value of the attribute
	 */
	protected Object value;
	
	protected Attribute(String name, int length, int type, Object value) {
		this.name = name;
		this.length = length;
		this.type = type;
		this.value = value;
	}
}
