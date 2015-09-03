Welcome to the C++ section of the CDMA documentation. It describes how to use the CDMA library to help the development of data analysis applications.

this section is divided into six sections:

  * an [Introduction](#Introduction.md) explaining the C++ port design and guidelines.

  * an section for data analysis application developers: [Client application development](CppClient.md).

  * an introduction section for contributors describing the [programmation guidelines](CppGuideLines.md) to follow when adding new functionalities to the core library or working on new or existing data format engines.

  * the [data format engine development](CppEngine.md) guide

  * the [Plug-in development](CppPlugin.md) guide

  * finally the automatically generated DOxygen documentation of the CDMA source code [API Reference guide](CppReference.md)


# Introduction #

The C++ implementation of the CDMA was developed after the Java version which was the first. Therefore this first version may lack some functionalities: the major functionality that is currently missing is the ability to write data.

This first C++ version was successfully compiled and tested on Linux and Windows 32-bits versions.

This implementation uses several modules implemented in a set of shared libraries ('.so' on Linux, '.dll' on Windows):

  * core library
  * engines libraries
  * data plug-ins

## Core library ##

The core library contains:

  * the `Factory` class: entry point for client applications,
  * a set of interfaces that must be implemented by the engines (may be override by plug-ins),
  * a set of tools for arrays manipulations
  * the dictionary mechanism,
  * the plug-ins system management (loading, method invocation).

The core library massively uses the YAT library, a lightweight generic cpp toolkit ([source code available here](http://tango-cs.svn.sourceforge.net/viewvc/tango-cs/share/yat/)).
In particular, it uses the smart pointers class `yat::SharedPtr` in order to avoid manual memory management and ensure code stability (less risk of memory leaks, object ownership troubles,...).

## Engines libraries ##

The engines are shared libraries that implement all the needed code to handle physical data formats. A data format engine can be used by several plug-ins (see below). They can't be directly used by the core library, but only through the plug-ins mechanism.

## Data plug-ins ##

Data plug-in encapsulates all the institute specific way of organizing data into physical containers (files, database, ...).

<i>To be completed...</i>

## Components schema ##

The schema below summarizes the dependencies between the different software components, from a application point of view.

![http://cdma.googlecode.com/svn/wiki/images/cpp_archi.png](http://cdma.googlecode.com/svn/wiki/images/cpp_archi.png)