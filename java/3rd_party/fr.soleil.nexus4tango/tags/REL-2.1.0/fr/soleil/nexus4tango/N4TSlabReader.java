package fr.soleil.nexus4tango;

import org.nexusformat.NexusException;

public class N4TSlabReader {

	/**
	 * @param args
	 */
	public static void main(String[] args) {

		// Slab definition
		// args[0] = file path
		// args[1] = node path in file (ex: "/{NXentry}/{NXdata}/my_data_set") 
		int[] start = { 5, 0, 0 }; // start position of slab must have same rank as dataset
		int[] shape = { 1, 494, 659 }; // shape of the slab must have same rank as dataset
		
		String file = "";
		String path = "";
		
		if( args.length > 0 ) {
			file = args[0];
		}
		
		if( args.length > 1 ) {
			path = args[1];
		}
		
		String[] groups = path.substring(0, path.lastIndexOf("/") ).split("/");
		String nodeName = path.substring(path.lastIndexOf("/") + 1 ); 
		
		PathData target = new PathData(groups, nodeName);
		
		NexusFileReader handler = new NexusFileReader(file);
		try {
			handler.openFile();

			handler.openPath(target);
			
			DataItem data = handler.readDataSlab(target, start, shape);
			
			Object values = data.getData();
			int i = 0;
			
			
			StringBuffer buf = new StringBuffer("shape: {");
			for( int value : shape ) {
				if( i != 0 ) {
					buf.append(", ");
				}
				buf.append(value);
				i++;
			}
			
			i = 0;
			buf.append("}\nstart: {\n");
			for( int value : start ) {
				if( i != 0 ) {
					buf.append(", ");
				}
				buf.append(value);
				i++;
			}
			
			i = 0;
			buf.append("}\nvalues: {");
			for( short value : (short[]) values) {
				if( i != 0 ) {
					if( i % shape[shape.length - 1] == 0) {
						buf.append("\nraw n°"+ (i / shape[shape.length - 1] ) + " = ");
					}
					else {
						buf.append(", ");
					}
				}
				buf.append(value);
				i++;
			}
			System.out.println(buf.toString());
			handler.closeFile();
		} catch (NexusException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
