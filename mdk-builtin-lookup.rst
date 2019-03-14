The MDK Built-in Lookup Framework
=================================

The MDK provides a way to implement a data- and configuration-driven keys lookup that doesn't require any custom code. This is done via a lookup class framework, contained in the ``oasislmf.model_preparation.lookup`` module, in which a point-polygon lookup for peril and a vulnerability dict. lookup for vulnerability can be implemented and integrated into a single lookup instance. This approach supports multiple perils and coverage types in the model.

1. Peril
--------

1.1. The peril lookup (``oasismf.model_preparation.lookup.OasisPerilLookup``) requires a pre-generated Rtree spatial index (http://toblerity.org/rtree/) which represents the underlying area peril grid for the model. The grid can be of a regular or variable resolution, and the cells in the grid represent the area polygons against which locations in the exposure are looked up for area peril IDs. Internally, the lookup of a given location is implemented via Rtree index queries, and the grid uses two approaches for the lookup: the first is to find a closest intersecting grid cell to the location, and the second, if no intersecting grid cell can be found, is to look for the closest grid cell to the location by Euclidean distance. Multiple results are sorted in order of ascending distance to the location, and the lookup always results in choosing the closest cell to the location

1.2. The MDK model CLI provides a command

::

    oasislmf model generate-peril-areas-rtree-file-index

to generate such an index from scratch from a source area peril dict. CSV file. The index is stored as a set of binary files, and it has a number of properties which be fine-tuned at the point of creation as well as when it is running.

1.3. The columns of the area peril CSV file must represent mappings of an area peril function, which must be a function of the area polygons, and peril and coverage type. For example, in an upcoming upgrade of the PiWind model, which will support the wind storm surge (``WSS``) and wind tropical
cyclone (``WTC``) perils, and buildings and contents coverage types (1, 3), the area peril dict. CSV file defines the columns

::

    PERIL_ID,COVERAGE_TYPE,LON1,LAT1,LON2,LAT2,LON3,LAT3,LON4,LAT4,AREA_PERIL_ID

where ``LON1,LAT1,...,LON4,LAT4`` are the columns for representing the longitude/latitudes of the four-sided underlying area polygons, ``PERIL_ID`` and ``COVERAGE_TYPE`` are the columns for the peril and coverage type combinations, and ``AREA_PERIL_ID`` is the area peril ID column

1.4. The underlying area peril function is viewed as the map

::

    (PERIL_ID, COVERAGE_TYPE, LON1, LAT1, ... , LON4, LAT4) |--> AREA_PERIL_ID

and the entries of the CSV file must be its mappings.

1.5. For example, for a given area, with corner points given by 

::

    -0.9176515,52.7339933,-0.9176515,52.74005981,-0.91158499,52.7339933,-0.91158499,52.74005981

the CSV file defines four mappings, one for each peril and coverage type
combination (``(WTC, 1)``, ``(WTC, 3)``, ``(WSS, 1)``, ``(WSS, 3)``), and the
corresponding area peril ID for that mapping. So for that area above the file
defines the entries

::

    PERIL_ID,COVERAGE_TYPE,LON1,LAT1,LON2,LAT2,LON3,LAT3,LON4,LAT4,AREA_PERIL_ID
    ...
    WTC,1,-0.9176515,52.7339933,-0.9176515,52.74005981,-0.91158499,52.7339933,-0.91158499,52.74005981,1
    WTC,3,-0.9176515,52.7339933,-0.9176515,52.74005981,-0.91158499,52.7339933,-0.91158499,52.74005981,1
    WSS,1,-0.9176515,52.7339933,-0.9176515,52.74005981,-0.91158499,52.7339933,-0.91158499,52.74005981,101
    WSS,3,-0.9176515,52.7339933,-0.9176515,52.74005981,-0.91158499,52.7339933,-0.91158499,52.74005981,101
    ...

Therefore, if your area peril grid contains ``10`` areas, and the model supports ``2`` perils and ``2`` coverage types then the area peril CSV file must define ``10 x 2 x 2 = 40`` mappings.

1.6. If your area peril ID function does not depend on cov. type then that is OK, but the lookup framework still requires coverage type to be specified in the CSV file, and all possible peril and coverage type mappings to be used in these mappings.

1.7. The area polygons need not be rectangles but can be ``n``-sided polygons, and different areas can have different numbers of edges, which is how the variable resolution grid property is supported. If for example, the maximum number of edges of your area polygons is ``10`` then
your CSV file must define 10 pairs of columns for 10 longitude/latitude corner points of the areas

::

    ...,LON1,LAT1,...,LON10,LAT10,...

But areas which, for example, are only ``4``-sided can be defined using the first four longitude/latitude corner point columns, e.g. ``LON1,LAT1,...,LON4,LAT4``, and the remaining column values for such areas can be left blank.

1.7. Once the area peril CSV file is complete then the properties of the file must be described in a JSON configuration file. Usually this is named ``lookup.json`` and can be kept with the model keys data. The `peril section of the PiWind lookup configuration file <https://github.com/OasisLMF/OasisPiWind/blob/kamdev/keys_data/PiWind/lookup.json#L12>`_ can be used as an example. The peril information is described in the subsection of the JSON named ``"peril"``, and it is more or
less self-explanatory.

1.8. The Rtree spatial index can be generated from the area peril CSV file using the command

::

    oasislmf model generate-peril-areas-rtree-file-index
           [-h] [-V] [-C CONFIG] [-c LOOKUP_CONFIG_FILE_PATH] [-d KEYS_DATA_PATH]
           [-f INDEX_FILE_PATH]

    Generates and writes an Rtree file index of peril area IDs (area peril IDs)
        and area polygon bounds from a peril areas (area peril) file.

    optional arguments:
      -h, --help            show this help message and exit
      -V, --verbose         Use verbose logging.
      -C CONFIG, --config CONFIG
                            MDK config. JSON file
      -c LOOKUP_CONFIG_FILE_PATH, --lookup-config-file-path LOOKUP_CONFIG_FILE_PATH
                            Lookup config file path
      -d KEYS_DATA_PATH, --keys-data-path KEYS_DATA_PATH
                            Keys data path
      -f INDEX_FILE_PATH, --index-file-path INDEX_FILE_PATH
                            Index file path (no file extension required)

The relevant flags here are ``-c`` for the lookup config. file path, ``-d`` for the model keys data path (which is usually where you would keep the lookup config. file path), and ``-f`` the intended filepath for the Rtree index files, including a filename prefix. Usually you would keep the Rtree index files as part of the model keys data. Once the index is generated, the index filename prefix should be set as the value of the ``"filename"`` key in the ``"rtree_index"`` subsection of the ``"peril"`` subsection of the lookup configuration file.

1.9. There is a difference with respect to how the index is created and stored when running the command in a Python ``2.7.6+`` environment and a Python ``3.x`` environment, because of differences in the way that index objects are pickled between Python ``2.7.6+`` and ``3.x``. The main thing to remember here is that Rtree indexes created in Python ``2.7.6+`` environments can also be read in Python ``3.x`` environments, but not vice versa. So for reasons of compatibility you may wish to create the index in a Python ``2.7.6+`` environment, but the MDK is compatible with both ``2.7.6+`` and ``3.x``.

1.10. The following iPython session excerpt for PiWind demonstrates how you would instantiate the built-in peril lookup provided a valid and complete lookup configuration file for your model.

::

   In [1]: import os

   In [2]: from oasislmf.utils.data import get_json, get_dataframe

   In [3]: from oasislmf.model_preparation.lookup import OasisLookupFactory as olf

   # Get the lookup config. dict from file
   In [4]: config = get_json('keys_data/PiWind/lookup.json')

   # Set the local model keys data path
   In [5]: config['keys_data_path'] = os.path.abspath('keys_data/PiWind')

   # Instantiate the lookup
   In [6]: _, plookup = olf.create(lookup_config=config, lookup_type='peril')

   In [7]: plookup
   Out[7]: <oasislmf.model_preparation.lookup.OasisPerilLookup at 0x110fdb978>

   # Inspect the areas Rtree index
   In [8]: idx = plookup.peril_areas_index

   In [9]: idx.bounds
   Out[9]: [-0.9176515, 52.7339933, -0.8569863999999999, 52.7946584]

   In [10]: idx.leaves()
   Out[10]: 
   [(0,
     [1,
      1,
      ...
      ...
      197],
     [-0.9176515, 52.7339933, -0.8569863999999999, 52.7946584])]

   # Inspect the model-supported peril IDs and coverage types
   In [11]: plookup.peril_ids
   Out[11]: ('WTC', 'WSS')

   In [12]: plookup.coverage_types
   Out[12]: (1, 3)

   # Inspect the expected column identifier of loc. IDs in the exposure
   In [13]: plookup.loc_id_col
   Out[13]: 'locnumber'

   # Define a location inside the index bounds and run a lookup against it
   # for the (`WSS`, 1) peril and coverage type combination
   In [14]: loc = {'locnumber': 1, 'longitude': -0.9176515, 'latitude': 52.7339933}

   In [15]: plookup.lookup({'locnumber': 1, 'longitude': -0.9176515, 'latitude': 52.7339933}, 'WSS', 1)
   Out[15]: 
   {'locnumber': 1,
    'longitude': -0.9176515,
    'latitude': 52.7339933,
    'peril_id': 'WSS',
    'coverage_type': 1,
    'status': 'success',
    'peril_area_id': 40,
    'area_peril_id': 40,
    'area_bounds': (-0.9176515, 52.7339933, -0.91158499, 52.74005981),
    'area_coordinates': ((-0.9176515, 52.7339933),
     (-0.9176515, 52.74005981),
     (-0.91158499, 52.74005981),
     (-0.91158499, 52.7339933),
     (-0.9176515, 52.7339933)),
    'message': 'Successful peril area lookup: 40'}

   # Run a lookup against a location outside the index
   In [16]: plookup.lookup({'locnumber': 1, 'longitude': -1.9176515, 'latitude': 52.7339933}, 'WSS', 1)
   Out[16]: 
   {'locnumber': 1,
    'longitude': -1.9176515,
    'latitude': 52.7339933,
    'peril_id': 'WSS',
    'coverage_type': 1,
    'status': 'fail',
    'peril_area_id': None,
    'area_peril_id': None,
    'area_bounds': None,
    'area_coordinates': None,
    'message': 'Peril area lookup: location is 1.0 units from the peril areas global boundary -  the required minimum distance is 0 units'}

1.11. Apart from the location-level lookup method (``lookup``), the peril lookup provides a bulk lookup method (``bulk_lookup``) which can accept a list, tuple, generator, pandas data frame or dict of location items, which can be dicts or Pandas series objects or any object which has as a dict-like interface.

2. Vulnerability
------------------

2.1. The vulnerability lookup (``oasismf.model_preparation.lookup.OasisVulnerabilityLookup``) is implemented via a simple Python dictionary that is built from a source vulnerability CSV file describing the vulnerability function. The vulnerability function is viewed as the map

::

    (PERIL_ID, COVERAGE_TYPE, **<LOC. PROPS>) |--> VULNERABILITY_ID

and the entries of the file must representing its mappings.

Here ``**<LOC PROPS>`` represents a sequence of columns representing loc. properties relevant for the vulnerability lookup for your model, including occupancy code, scheme, building class, etc. The column names pertaining to the location properties should be OED-compatible, e.g. ``OccupancyCode``. The `vulnerability section of the PiWind lookup configuration file <https://github.com/OasisLMF/OasisPiWind/blob/kamdev/keys_data/PiWind/lookup.json#L71>`_ can be used as an example.

2.2. The following iPython session excerpt for PiWind demonstrates how you would instantiate the built-in vulnerability lookup provided a valid and complete lookup configuration file for your model.

::

   # Instantiate the vuln. lookup
   In [17]: _, vlookup = olf.create(lookup_config=config, lookup_type='vulnerability')

   In [18]: vlookup
   Out[18]: <oasislmf.model_preparation.lookup.OasisVulnerabilityLookup at 0x111127e10>

   # Inspect the vulnerability function dict. - should match the vuln. dict CSV file
   In [19]: vlookup.vulnerabilities
   Out[19]: 
   OrderedDict([(('WTC', 1, 1000), 1),
   ...
   ...
   , 3031), 12)])

   # Inspect the vuln. section of the lookup config. (can also be done from the config.)
   In [20]: vlookup.config['vulnerability']
   Out[20]: 
   {'file_path': '/path/to/OasisPiWind/keys_data/PiWind/vulnerability_dictOED3.csv',
    'file_type': 'csv',
    'float_precision_high': True,
    'num_vulnerabilities': 684,
    'cols': ('PERIL_ID', 'COVERAGE_TYPE', 'OCCUPANCYCODE', 'VULNERABILITY_ID'),
    'non_na_cols': ('PERIL_ID',
     'COVERAGE_TYPE',
     'OCCUPANCYCODE',
     'VULNERABILITY_ID'),
    'key_cols': ('PERIL_ID', 'COVERAGE_TYPE', 'OCCUPANCYCODE'),
    'col_dtypes': {'PERIL_ID': 'str',
     'COVERAGE_TYPE': 'int',
     'OCCUPANCYCODE': 'int',
     'VULNERABILITY_ID': 'int'},
    'sort_col': 'vulnerability_id',
    'sort_ascending': True,
    'vulnerability_id_col': 'vulnerability_id'}

   # Inspect the peril and coverage types (can also be done from the config.)
   In [21]: vlookup.peril_ids
   Out[21]: ('WTC', 'WSS')

   In [22]: vlookup.coverage_types
   Out[22]: (1, 3)

   # Define a location with the properties required for the vuln. function
   In [23]: loc = {'locnumber': 1, 'occupancycode': 1000}

   In [24]: vlookup.lookup(loc, 'WSS', 1)
   Out[24]: 
   {'locnumber': 1,
    'peril_id': 'WSS',
    'coverage_type': 1,
    'status': 'success',
    'vulnerability_id': 7,
    'message': 'Successful vulnerability lookup: 7',
    'occupancycode': 1000}

2.3. Apart from the location-level lookup method (``lookup``), the vulnerability lookup, like the peril lookup, provides a bulk lookup method (``bulk_lookup``) which can accept a list, tuple, generator, pandas data frame or dict of location items, which can be dicts or Pandas series objects or any object which has as a dict-like interface.

3. Integration and Configuration
----------------------------------

3.1. The integrated lookup (``oasismf.model_preparation.lookup.OasisLookup``) combines the peril and vulnerability lookups, and contains them as attributes. It provides a location-level lookup accepting a location dict. or Pandas series, and a peril and coverage type combination, as well as a bulk lookup that accepts a iterable sequence of location dicts. or Pandas series. The relationship between the peril, vulnerabilty and combined lookups can be seen in the following class diagram.

    .. image:: mdk-builtin-lookup-class-framework.jpg

3.2. The properties of the integrated lookup and also other properties such as the keys data path, and model-supported coverage types must be defined in the appropriate sections of the lookup configuration file. Again, the `PiWind lookup configuration file <https://github.com/OasisLMF/OasisPiWind/blob/kamdev/keys_data/PiWind/lookup.json>`_ is a good example to refer to.

3.3. One point to note is that the lookup configuration file also defines a section to define the properties of the input exposure, named ``"exposure"``. For PiWind it looks like this

::

    "exposure": {
        "id_col": "LocNumber",
        "coords_type": "lonlat",
        "coords_x_col": "Longitude",
        "coords_y_col": "Latitude",
        "coords_x_bounds": [-180, 180],
        "coords_y_bounds": [-90, 90],
        "non_na_cols": ["LocNumber", "Longitude", "Latitude"],
        "col_dtypes": {
            "LocNumber": "str", "Longitude": "float", "Latitude": "float"
        },
        "sort_col": "LocNumber",
        "sort_ascending": true
    }

These properties basically describe certain key columns in the source exposure file, which should be an OED-compatible file.

3.4. The following iPython session excerpt for PiWind demonstrates how you would instantiate the built-in combined lookup provided a valid and complete lookup configuration file for your model.

::

   In [25]: _, lookup = olf.create(lookup_config=config)

   In [26]: lookup
   Out[26]: <oasislmf.model_preparation.lookup.OasisLookup at 0x11168e0b8>

   In [27]: loc = {'locnumber': 1, 'longitude': -0.9176515, 'latitude': 52.7339933, 'occupancycode': 1000}

   In [28]: lookup.lookup(loc, 'WTC', 1)
   Out[28]: 
   {'locnumber': 1,
    'peril_id': 'WTC',
    'coverage_type': 1,
    'area_peril_id': 40,
    'vulnerability_id': 1,
    'status': 'success',
    'message': 'Successful peril area lookup: 40; Successful vulnerability lookup: 1',
    'occupancycode': 1000}

    # Get the peril lookup from the combined lookup
    In [29]: _plookup = lookup.peril_lookup

    In [30]: _plookup
    Out[30]: <oasislmf.model_preparation.lookup.OasisPerilLookup at 0x11168efd0>

    # Get the vuln. lookup from the combined lookup
    In [31]: _vlookup = lookup.vulnerability_lookup

    In [32]: _vlookup
    Out[32]: <oasislmf.model_preparation.lookup.OasisVulnerabilityLookup at 0x11168ea90>

    # Get the full combined config.
    In [33]: config = lookup.config

    In [34]: config
    Out [34]:
    {'model': {'supplier_id': 'OasisLMF',
    'model_id': 'PiWind',
    'model_version': '0.0.0.1'},
    'keys_data_path': '/path/to/OasisPiWind/keys_data/PiWind',
    'coverage': {'coverage_types': [1, 3], 'coverage_type_col': 'COVERAGE_TYPE'},
    'peril': {'peril_ids': ['WTC', 'WSS'],
    ...
    ...
    'vulnerability': {'file_path': '/path/to/OasisPiWind/keys_data/PiWind/vulnerability_dictOED3.csv',
    ...
    ...
    'exposure': {'id_col': 'LocNumber',
    ...
    ...
    'sort_ascending': True}}


3.5. Like the peril and vulnerability lookups, the combined lookup also provides a bulk lookup method (``bulk_lookup``) which can accept a list, tuple, generator, pandas data frame or dict of location items, which can be dicts or Pandas series objects or any object which has as a dict-like interface.
