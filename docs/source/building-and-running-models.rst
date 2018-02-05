Building and Running Models
===========================

The repository provides a Python toolkit for building, running and
testing Oasis models end-to-end, including performing individual steps
in this process. It includes:

-  a Python class framework for working with Oasis models and model
   resources as Python objects (the ``models`` subpackage)
-  a Python class framework for managing model exposures and resources,
   and also for generating Oasis files from these (the ``exposures``
   subpackage)
-  a Python factory class for instantiating keys lookup services for
   models, and generating and saving keys outputs from these lookup
   services (the ``keys`` subpackage)
-  executable scripts, based on these class frameworks, for writing keys
   outputs from model lookup services (``generate_keys.py``), generating
   Oasis files from model source exposures and other resources
   (``generate_oasis_files.py``), and generating losses for models
   (``generate_losses.py``). This includes a "master" script that can
   perform all these steps to run the model end-to-end
   (``run_model.py``).

Generating keys
---------------

``generate_keys.py`` is an executable script which can generate and
write Oasis keys (area peril ID, vulnerability ID) for a model, given
the following arguments (in no particular order)

::

    ./generate_keys.py -k /path/to/keys/data
                       -v /path/to/model/version/csv/file
                       -l /path/to/lookup/service/package
                       -e /path/to/model/exposures/csv/file
                       -o /path/to/output/file
                       [-f <output format - 'oasis_keys' or 'list_keys'>]

When calling the script this way paths can be given relative to the
script, in particular, file paths should include the filename and
extension. The paths to the keys data, lookup service package (Python
package containing the lookup source code), and model version file will
usually be located in the model keys server (Git) repository. If the
repository was created by or is managed by Oasis LMF then the lookup
service package will usually be contained in the ``src/keys_server``
Python subpackage and can be given as the path to that subpackage (see
the OasisPiWind repository as a reference for how to structure an Oasis
keys server repository)

It is also possible to run the script by defining these arguments in a
JSON configuration file and calling the script with option ``-f`` and
the (relative or absolute) path to the file.

::

    ./generate_keys.py -f /path/to/script/config/json/file

The JSON file should contain the following keys (in no particular order)

::

    "keys_data_path"
    "model_version_file_path"
    "lookup_package_path"
    "model_exposures_file_path"
    "output_file_path"
    "output_format"

and the values of the path-related keys should be string paths, given
relative to the location of JSON file. The JSON file is usually placed
in the model keys server repository. The ``"output_format"`` key is
optional - by default the script will generate an Oasis keys file.

Keys records returned by an Oasis keys lookup service (see the PiWind
lookup service for reference) will be Python dicts with the following
structure

::

    {
        "id": <loc. ID>,
        "peril_id": <Oasis peril type ID - oasis_utils/oasis_utils.py>,
        "coverage": <Oasis coverage type ID - see oasis_utils/oasis_utils.py>,
        "area_peril_id": <area peril ID>,
        "vulnerability_id": <vulnerability ID>,
        "message": <lookup status message>,
        "status": <lookup status code - see oasis_utils/oasis_utils.py>
    }

The ``generate_keys.py`` script can generate keys records in this
format, and write them to file.

For model loss calculations however ktools requires a keys CSV file with
the following format

::

    LocID,PerilID,CoverageID,AreaPerilID,VulnerabilityID
    ..
    ..

where the headers correspond to the relevant Oasis keys record fields. The ``generate_keys.py`` script can also generate and
write Oasis keys files.

Generating Oasis files
----------------------

``generate_oasis_files.py`` is an executable script which can generate
Oasis files (items, coverages, GUL summary) for a model, given the
following arguments (in no particular order)

::

    ./generate_oasis_files.py -k /path/to/keys/data
                              -v /path/to/model/version/csv/file
                              -l /path/to/lookup/service/package
                              -p /path/to/canonical/exposures/profile/JSON/file
                              -e /path/to/source/exposures/file
                              -a /path/to/source/exposures/validation/file
                              -b /path/to/source/to/canonical/exposures/transformation/file
                              -c /path/to/canonical/exposures/validation/file
                              -d /path/to/canonical/to/model/exposures/transformation/file
                              -x /path/to/xtrans/executable
                              [-o /path/to/oasis/files/directory]

When calling the script this way paths can be given relative to the
script, in particular, file paths should include the filename and
extension. The paths to the keys data, lookup service package, model
version file, canonical exposures profile JSON, source exposures file,
transformation and validation files, will usually be located in the
model keys server repository. The path to the Oasis files directory is
optional - by default the script will create a timestamped folder in
``omdk/runs`` with the prefix ``OasisFiles``.

It is also possible to run the script by defining these arguments in a
JSON configuration file and calling the script using the path to this
file using the option ``-f`` and the (relative or absolute) path to the
file.

::

    ./generate_oasis_files.py -f /path/to/script/config/json/file

The JSON file contain the following keys (in no particular order)

::

    "keys_data_path"
    "model_version_file_path"
    "lookup_package_path"
    "canonical_exposures_profile_json_path"
    "source_exposures_file_path"
    "source_exposures_validation_file_path"
    "source_to_canonical_exposures_transformation_file_path"
    "canonical_exposures_validation_file_path"
    "canonical_to_model_exposures_transformation_file_path"
    "xtrans_path"
    "oasis_files_path"

and the values of these keys should be string paths, given relative to
the location of the JSON file. The JSON file is usually placed in the
model keys server repository. The ``"oasis_files_path"`` key is optional
- by default the script will create a timestamped folder in
``omdk/runs`` with the prefix ``OasisFiles``.

Generating losses
-----------------

``generate_losses.py`` is an executable script which, given Oasis files,
model analysis settings JSON file, model data, and some other
parameters, can generate losses using the installed ktools framework.
The script can be called directly from the command line given the
following arguments (in no particular order)

::

    ./generate_losses.py -o /path/to/oasis/files
                         -j /path/to/analysis/settings/json/file
                         -m /path/to/model/data
                         [-r /path/to/model/run/directory]
                         [-s <ktools script name (without file extension)>]
                         [-n <number of ktools calculation processes to use>]
                         [--execute | --no-execute]

When calling the script this way paths can be given relative to the
script, in particular, file paths should include the filename and
extension. The path to the model run directory is optional - by default
the script will create a timestamped folder in ``omdk/runs`` with the
prefix ``ProgOasis``. The ktools script name and number of calculation
processes are optional - by default the script will create a ktools
script named ``run_tools.sh`` and set the number of calculation
processes to 2. By default executing ``generate_losses.py`` will
automatically execute the ktools losses script it generates. If you
don't want this provide the (optional) ``--no-execute`` argument. The
default here is automatic execution.

The script copies the analysis settings JSON file to the model run
directory and sets up the following folder structure inside

::

    ├── analysis_settings.json
    ├── fifo/
    ├── input/
    ├── output/
    ├── static/
    └── work/

Depending on the OS type the model data is symlinked (Linux, Darwin) or
copied (Cygwin, Windows) into the ``static`` subfolder. The input files
are kept in the ``input`` subfolder and the losses are generated as CSV
files in the ``output`` subfolder.

It is also possible to run the script by defining these arguments in a
JSON configuration file and calling the script using the path to this
file using the option ``-f`` and the (relative or absolute) path to the
file.

::

    ./generate_losses.py -f /path/to/script/config/json/file'

The JSON file should contain the following keys (in no particular order)

::

    "oasis_files_path"
    "analysis_settings_json_file_path"
    "model_data_path"
    "model_run_dir_path"
    "ktools_script_name"
    "ktools_num_processes"
    "execute"

and the values of the path-related keys should be string paths, given
relative to the location of the JSON file. The JSON file is usually
placed in the model keys server repository. The ``"model_run_dir_path"``
key is optional - by default the script will create a timestamped folder
in ``omdk/runs`` with the prefix ``ProgOasis``. The
``"ktools_script_name"`` and ``"ktools_num_processes"`` keys are
optional - by default the script will create a ktools script named
``run_tools.sh`` and set the number of calculation processes to 2. The
``"execute"`` key is optional - if present it should be either ``true``
or ``false`` depending on whether you want the generated ktools losses
scripts to be automatically executed or not. The default here is
automatic execution.

Running a model end-to-end
--------------------------

``run_model.py`` is an executable "master" script that can run models
end-to-end, i.e. generate losses given model resources, including keys
data, canonical exposure profiles, exposure transformation and
validation files, model data, analysis settings etc., given the
following arguments (in no particular order)

::

    ./run_model.py -k /path/to/keys/data/folder
                   -v /path/to/model/version/file
                   -l /path/to/lookup/service/package
                   -p /path/to/canonical/exposures/profile/JSON/file
                   -e /path/to/source/exposures/file
                   -a /path/to/source/exposures/validation/file
                   -b /path/to/source/to/canonical/exposures/transformation/file
                   -c /path/to/canonical/exposures/validation/file
                   -d /path/to/canonical/to/model/exposures/transformation/file
                   -x /path/to/xtrans/executable
                   -j /path/to/analysis/settings/json/file
                   -m /path/to/model/data
                   [-r /path/to/model/run/directory]
                   [-s <ktools script name (without file extension)>]
                   [-n <number of ktools calculation processes to use>]

When calling the script this way paths can be given relative to the
script, in particular, file paths should include the filename and
extension. The paths to the keys data, lookup service package, model
version file, canonical exposures profile JSON, source exposures file,
transformation and validation files, and analysis settings JSON file,
will usually be located in the model keys server repository. The path to
the model run directory is optional - by default the script will create
a timestamped folder in ``omdk/runs`` with the prefix ``ProgOasis``. The
ktools script name and number of calculation processes are also optional
- by default the script will create a ktools script named
``run_tools.sh`` and set the number of calculation processes to 2.

It is also possible to run the script by defining these arguments in a
JSON configuration file and calling the script using the path to this
file using the option ``-f`` and the (relative or absolute) path to the
file.

::

    ./run_model.py -f /path/to/script/config/json/file'

The JSON file should contain the following keys (in no particular order)

::

    "keys_data_path"
    "model_version_file_path"
    "lookup_package_path"
    "canonical_exposures_profile_json_path"
    "source_exposures_file_path"
    "source_exposures_validation_file_path"
    "source_to_canonical_exposures_transformation_file_path"
    "canonical_exposures_validation_file_path"
    "canonical_to_model_exposures_transformation_file_path"
    "xtrans_path"
    "analysis_settings_json_file_path"
    "model_data_path"
    "model_run_dir_path"
    "ktools_script_name"
    "ktools_num_processes"

and the values of the path-related keys should be string paths, given
relative to the location of the JSON file. The JSON file is usually placed in the model keys server
repository. The ``"model_run_dir_path"`` key is optional - by default
the script will create a timestamped folder in ``omdk/runs`` with the
prefix ``ProgOasis``. The ``"ktools_script_name"`` and
``"ktools_num_processes"`` keys are optional - by default the script
will create a ktools script named ``run_tools.sh`` and set the number of
calculation processes to 2.

You can define a separate JSON configuration file for each model,
provided you have the model keys server repository and other required
model resources available locally.

**NOTE**: For a given model the JSON script configuration files for
``generate_oasis_files.py``, ``generate_losses.py`` and ``run_model.py``
should complement each other, except for ``generate_losses.py`` which
requires the path to Oasis files, not required by ``run_model.py``. You
can run any of these scripts against a single master script
configuration file, provided that the path to an actual set of Oasis
files is added in order to run ``generate_losses.py``.

Running PiWind
--------------

PiWind is a reference windstorm model developed by Oasis. The lookup
source code, keys server, keys data and model data and all other model
resources are in the model GitHub repository which is

https://github.com/OasisLMF/OasisPiWind

The repository also contains a `JSON configuration file <ttps://github.com/OasisLMF/OasisPiWind/blob/master/mdk-oasislmf-piwind.json>`_ for the model
which can be used to run it end-to-end with the MDK master script.

::

    {
        "keys_data_path": "keys_data/PiWind",
        "model_version_file_path": "keys_data/PiWind/ModelVersion.csv", 
        "lookup_package_path": "src/keys_server",
        "canonical_exposures_profile_json_path": "oasislmf-piwind-canonical-profile.json",
        "source_exposures_file_path": "tests/data/SourceLocPiWind.csv",
        "source_exposures_validation_file_path": "flamingo/PiWind/Files/ValidationFiles/Generic_Windstorm_SourceLoc.xsd",
        "source_to_canonical_exposures_transformation_file_path": "flamingo/PiWind/Files/TransformationFiles/MappingMapToGeneric_Windstorm_CanLoc_A.xslt",
        "canonical_exposures_validation_file_path": "flamingo/PiWind/Files/ValidationFiles/Generic_Windstorm_CanLoc_B.xsd",
        "canonical_to_model_exposures_transformation_file_path": "flamingo/PiWind/Files/TransformationFiles/MappingMapTopiwind_modelloc.xslt",
        "xtrans_path": "../omdk/xtrans/xtrans.exe",
        "analysis_settings_json_file_path": "analysis_settings.json",
        "model_data_path": "model_data/PiWind"
    }

**NOTE**: All the paths, except for the ``xtrans.exe`` executable, are
given relative to the location of the PiWind repository. The
``xtrans.exe`` executable is not part of the MDK repository, and you
need to build it for your platform by running the ``make-trans``
executable shell script - this will built it in the ``omdk/xtrans``
folder. For the ``xtrans.exe`` path to be found you should locate the
MDK repository adjacent to the PiWind repository, i.e.

::

    ...
    |- OasisPiWind/
    |- omdk/
    ...
