# Namespaces #

For the time being there are not too many plugins and engines available for CDMA. Consequently the number of classes implementing various interfaces provided by abstract classes in the CDMA core implementation is rather low and it might be convenient to use prefixes to identify the various classes of a particular engine or plugin. However with an increasing number of file formats supported and plugins being developed such an approach becomes difficult to manage.

To solve this problem I suggest to sort the different components into a couple of namespaces with the following layout:
  * all core components go to `cdma::core`
  * engines below `cdma::engines::<enginename>`
  * and plugins to `cdma::plugins::<pluginname>`
Thus, the components of an engine reading `SPEC` files would reside below `cdma::engines::spec` and those of a plugin for DESY data will be available under `cdma::plugins::desy`. A plugin or engine developer must only find a unique name for the plugin or engine which will be used as a namespace. This makes an additional coding guideline possible: all implementations of interfaces provided by a plugin or engine have the same name as the interface without the leading `I` (`IDataItem` will become `DataItem`). As every plugin and engine has its own namespace no conflicts with class names can occur. This would make it easier for other developers to find the interface implementations in the code of a particular engine or plugin.

# Packaging #

The success of a project not only depends on how well the code works but also on how easy it is to get it installed. In other words installation packages should be provided for those operating system platforms we want to support. Packaging is already a tedious job with well designed code but it can become a real pain if the developers of the software did not spend  a minute on thinking how their software will be installed on a system.
It is thus a good idea to configure the build system in a way that makes life for package-maintainers as easy as possible.

## Installation directories ##
Let us assume that all the code will be installed below a directory provided by the environment variable `PREFIX` (on Linux system the default will be `PREFIX=/usr`). The code should than be installed as follows

| **directory** | **installed component** |
|:--------------|:------------------------|
| `PREFIX/include/cdma` | header files of the CDMA core library |
| `PREFIX/include/cdma/engines/<enginenname>` | header files for the engine `enginename` |
| `PREFIX/include/cdma/plugins/<pluginname>` | header files for the plugin `pluginname` |
| `PREFIX/lib`  | shared objects of the core library and shared objects of the engines |
| `PREFIX/lib/plugins` | loadable modules for plugins |
| `/etc/cdma`   | runtime configuration on Linux. On Windows systems this might be done via the Windows registry or the information can be stored below `PREFIX/etc/cdma` |
It is important to note that this scheme must not only be kept by CDMA core developers but also by each engine or plugin developer.

## Suggested package structure ##

On Windows there is usually only one package per component while on RPM or DEB based Linux installations typically three packages per component are required (or at least this is the way how it SHOULD be done).
| **component** | **Windows package** | **RPM and DEB packages on Linux**|
|:--------------|:--------------------|:---------------------------------|
| CDMA core     | `CDMACore`          | `libcdmacore`, `libcdmacore-dev`, `libcdmacore-doc` |
| Nexus engine  | `NexusEngine`       | `libcdmaengine-nexus`,`libcdmaengine-nexus-dev`, `libcdmaengine-nexus-doc` |
| Nexus plugin  | `NexusPlugin`       | `libcdmaplugin-nexus`, `libcdmaplugin-nexus-dev`, `libcdmaplugin-nexus-doc`|

# Documentation #

Although the API documentation seems to be complete I strongly recommend to add additional information in particular about how objects are related. For instance a document which explains how Array, ArrayStorage and views are related to each other. Or some documentation explaining what what "shared dimension" are.

# CDMA design #

Some things in the design of the C++ API for CDMA which are worthy to discuss.

## Interfaces ##

Some remarks on various interfaces.

### Provide default implementations ###
The interfaces provided by CDMA core are pure abstract classes (most probably this is some relic from the JAVA interfaces). In other words developers writing engines and plugins have to implement all the methods exposed by the interfaces. This causes a lot of tedious typing work if only a few of the methods are used. In particular all methods related to writing data are actually useless and a default implementation would be easy to do by just throwing a NotImplemented exception.
In short: provide default implementations even if they only throw a NotImplemented exception.

### Shared Pointer problem ###

CDMA makes heavy use of shared pointers. Although smart pointers (in particular shared pointers) are a good choice to avoid ownership problems they cause certain complications if they refer to objects where shared ownership is not a good choice. This is for instance true for stream objects. There are good reasons why streams cannot be copied in C++ - one of them is the stream positions. In particular for streams unique\_ptr might be the better choice. It is in general no good idea to have several objects referring to the same IO object (except maybe for HDF5/Nexus) and it is particularly bad if one thinks about thread-safety (at least to my opinion).