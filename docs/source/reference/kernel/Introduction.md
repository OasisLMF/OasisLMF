# 1. Introduction
The Oasis calculation **kernel** is a “specification” of a set of streaming processes —
it defines the processing architecture and binary data structures — implemented as a set
of components (the “reference model”) that can be adapted for particular model or business
needs. The kernel is now implemented in Python (**pytools**); it was historically the C++
toolkit “ktools”, which is being decommissioned.

Installation instructions can be found in {doc}`the installation guide </installation>`.

### Background
The Kernel performs the core Oasis calculations of computing effective damageability distributions, Monte-Carlo sampling of ground up loss, the financial module calculations, which apply insurance policy terms and conditions to the sampled losses, and finally some common catastrophe model outputs.

The early releases of Oasis used a SQL-compliant database to perform all calculations.  Release 1.3  included the first “in-memory” version of the Oasis Kernel written in C++ and C to provide streamed calculation at high computational performance, as an alternative to the database calculation. The scope of the in-memory calculation was for the most intensive Kernel calculations of ground up loss sampling and the financial module. This in-memory variant was first delivered as a stand-alone toolkit "ktools" with R1.4. 

With R1.5, a Linux-based in-memory calculation back-end was released, using the reference model components of ktools. The range of functionality of ktools was still limited to ground up loss sampling, the financial module and single output workflows, with effective damage distributions and output calculations still being performed in a SQL-compliant database.

In 2016 the functionality of ktools was extended to include the full range of Kernel calculations, including effective damageability distribution calculations and a wider range of financial module and output calculations.  The data stream structures and input data formats were also substantially revised to handle multi-peril models, user-definable summary levels for outputs, and multiple output workflows.  

In 2018 the Financial Module was extended to perform net loss calculations for per occurrence forms of reinsurance, including facultative reinsurance, quota share, surplus share, per risk and catastrophe excess of loss treaties.

### Architecture

The Kernel is provided as a toolkit of components (“ktools”) which can be invoked at the command line.  Each component is a separately compiled executable with a binary data stream of inputs and/or outputs.

The principle is to stream data through the calculations in memory, starting with generating the damage distributions and ending with calculating the user's required result, before writing the output to disk.  This is done on an event-by-event basis, which means at any one time the compute server only has to hold the model data for a single event in its memory, per process. The user can run the calculation across multiple processes in parallel, specifiying the analysis workfkow and number of processes in a script file appropriate to the operating system.

### Language

The components can be written in any language as long as the data structures of the binary streams are adhered to.  The current set of components have been written in POSIX-compliant C++.  This means that they can be compiled in Linux and Windows using the latest GNU compiler toolchain.

### Components

The components in the Reference Model can be summarized as follows;

* **[Core components](CoreComponents.md)** perform the core kernel calculations and are used in the main analysis workflow.
* **[Output components](OutputComponents.md)** perform specific output calculations at the end of the analysis workflow. Essentially, they produce various statistical summaries of the sampled losses from the core calculation.
* **[Data conversion components](DataConversionComponents.md)** are provided to enable users to convert model and exposure data into the required binary formats.
* **[Stream conversion components](StreamConversionComponents.md)** are provided to inspect the data streams flowing between the core components, for more detailed analysis or debugging purposes.
 
### Usage

Standard piping syntax can be used to invoke the components at the command line. It is the same syntax in Windows DOS, Linux terminal or Cygwin (a Linux emulator for Windows). For example the following command invokes evepy, modelpy, gulmc, fmpy, summarypy and eltpy, and exports an event loss table output to a csv file.

``` sh
$ evepy 1 1 | modelpy | gulmc -r –S100 -a1 –i - | fmpy | summarypy -f -1 - | eltpy > elt.csv
```

Example python scripts are provided along with a binary data package in the /examples folder to demonstrate usage of the toolkit. For more guidance on how to use the toolkit, see [Workflows](Workflows.md).

[Go to 2. Data streaming architecture overview](Overview.md)

[Back to Contents](Contents.md)
