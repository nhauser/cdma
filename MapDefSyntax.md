# Introduction #

The **mapping document for a data source belongs to a plug-in**. It is its specific way of getting and constructing a data item. Each data source's physical measurement should have a mapping entry. Each mapping entry should subscribe to a concept (whether in the global concepts document or in the one dedicated to an experiment). Therefore the resulting data item should conform to the concept it corresponds.

The fact that **mapping conforms to concepts** is under the **plug-in's responsibility**.

A plug-in managing various experiment of different data source structure can have various mapping file (see [developing a plug-in](JavaPlugin#Customize_Extended_Dictionary_mechanism.md) ).

A plug-in's mapping file should be (but not inevitably) the same one for a Java or a C++ plug-in of the same data source.

All mappings files for a specific plug-in are searched in [dictionaries folder](http://code.google.com/p/cdma/source/browse/dictionaries/trunk) under the institute's plug-in ID.

# Mapping principles #

The mapping document is an XML file, analysed by the _ExtendedDictionary_ at its loading. That's why it must respect a given DTD ( see [concepts\_file.dtd](http://code.google.com/p/cdma/source/browse/dictionaries/trunk/concepts_file.dtd) ).

In the mapping file, the plug-in developer specifies things to do for each entry. The aim is to construct a data item fitting the concept it corresponds to.

For each entry, to obtain the data item, there are mainly 2 steps:
  * the construction of the data item (name, shape, type, location, matrix...)
  * the construction of its attributes

For each step of the item construction, there are 2 commands (declared using a XML mark-up) available in mapping (that can be repeated as many times as requested):
  * the opening of a _path_ in the data source
  * the execution at runtime of a specific method (see [runtime code execution](JavaPlugin#Customize_runtime_code_execution.md))

While constructing the item, the result of each executed command is re-injected as in the context of the next execution.

Note: **all mark-ups are interpreted by the Extended Dictionary Mechanism, _but_ not their contents which are only considered by the plug-in.**

# Mapping syntax #

Below are some samples of the mapping syntax in the case of nexus SOLEIL's plug-in.

The case of _comments_, here we only need to return the content of the node and no need for specific things to be done. So the mapping is a simple path to be opened:
```
  <item key="conditions_comments">
    <path>/{NXentry}/comment_conditions/data</path>
  </item>
```

The case of a _total\_image_, here we want to set an attribute containing the acquisition sequence's duration name (that is situated on another node):
```
  <item key="total_image">
    <path>/{NXentry}/Scienta_Total{NXdata}/total_data</path>
    <attribute name="duration">
        <path>/{NXentry}/duration</path>
    </attribute>
  </item>
```

The _samplePhi_ case, the measurement have been split among several nodes, so here we will call a method which aims to stack them into a virtual single one, according their names:
```
  <item key="samplePhi">
    <path>/{NXentry}/{NXdata}/phi*</path>
    <call>DataItemStackerByName</call>
  </item>
```

The case of _spectrums\_xia_ mapping. It is the most complicated we have. Indeed we want to return a data item, with some attributes _equipement_, region\_of\_equipment_. But all those informations do not belong to the node containing data. More over data is split among several nodes in the data source. So here we have to gather data into a big data item, harvest attributes using a generic method, and finally add a node by hand called_equipment