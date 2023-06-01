

MODELS_TABLE = """+------+---------------+------------+--------------+
|   id | supplier_id   | model_id   |   version_id |
|------+---------------+------------+--------------|
|    1 | OasisLMF      | PiWind     |            1 |
|    2 | OasisLMF      | PiQuake    |            3 |
+------+---------------+------------+--------------+"""

PORT_TABLE = """+------+---------------------------+-----------------+-----------------+-------------------------+--------------------------+
|   id | name                      | location_file   | accounts_file   | reinsurance_info_file   | reinsurance_scope_file   |
|------+---------------------------+-----------------+-----------------+-------------------------+--------------------------|
|    1 | Portfolio_31052023-145540 | Yes             | Yes             | Yes                     | Yes                      |
|    2 | Portfolio_31052023-150223 | Yes             | Yes             | -                       | -                        |
+------+---------------------------+-----------------+-----------------+-------------------------+--------------------------+"""

ANAL_TABLE = """+------+--------------------------+---------+-------------+-------------------------+--------------+---------------+----------------+
|   id | name                     |   model |   portfolio | status                  | input_file   | output_file   | run_log_file   |
|------+--------------------------+---------+-------------+-------------------------+--------------+---------------+----------------|
|    4 | Analysis_31052023-154122 |       1 |           1 | INPUTS_GENERATION_ERROR | -            | -             | -              |
|    6 | Analysis_31052023-155134 |       1 |           1 | READY                   | Linked       | -             | -              |
|    7 | Analysis_31052023-155535 |       2 |           2 | RUN_COMPLETED           | Linked       | Linked        | Linked         |
+------+--------------------------+---------+-------------+-------------------------+--------------+---------------+----------------+"""


RETURN_MODELS = [
 {
   "id": 1,
   "supplier_id": "OasisLMF",
   "model_id": "PiWind",
   "version_id": "1",
   "created": "2023-05-31T13:55:27.244296Z",
   "modified": "2023-06-01T07:34:27.901419Z",
   "data_files": [],
   "resource_file": "http://localhost:8000/v1/models/1/resource_file/",
   "settings": "http://localhost:8000/v1/models/1/settings/",
   "versions": "http://localhost:8000/v1/models/1/versions/"
 },
 {
   "id": 2,
   "supplier_id": "OasisLMF",
   "model_id": "PiQuake",
   "version_id": "3",
   "created": "2023-05-31T13:55:27.244296Z",
   "modified": "2023-06-01T07:34:27.901419Z",
   "data_files": [],
   "resource_file": "http://localhost:8000/v1/models/1/resource_file/",
   "settings": "http://localhost:8000/v1/models/1/settings/",
   "versions": "http://localhost:8000/v1/models/1/versions/"
 },
]


RETURN_PORT = [
  {
    "id": 1,
    "name": "Portfolio_31052023-145540",
    "created": "2023-05-31T13:55:40.411212Z",
    "modified": "2023-05-31T13:55:40.736394Z",
    "accounts_file": {
      "uri": "http://localhost:8000/v1/portfolios/1/accounts_file/",
      "name": "SourceAccOEDPiWind.csv",
      "stored": "3e0e65f1de534c66b425e0cd406f1fda.csv"
    },
    "location_file": {
      "uri": "http://localhost:8000/v1/portfolios/1/location_file/",
      "name": "SourceLocOEDPiWind10.csv",
      "stored": "669c6ee20e33490f9e572dfe726dadab.csv"
    },
    "reinsurance_info_file": {
      "uri": "http://localhost:8000/v1/portfolios/1/reinsurance_info_file/",
      "name": "SourceReinsInfoOEDPiWind.csv",
      "stored": "d1c9c1fd05594f5ba6164f2e27f6a136.csv"
    },
    "reinsurance_scope_file": {
      "uri": "http://localhost:8000/v1/portfolios/1/reinsurance_scope_file/",
      "name": "SourceReinsScopeOEDPiWind.csv",
      "stored": "93521b6c599c4882878d61fdfe7f207d.csv"
    },
    "storage_links": "http://localhost:8000/v1/portfolios/1/storage_links/",
    "groups": []
  },
  {
    "id": 2,
    "name": "Portfolio_31052023-150223",
    "created": "2023-05-31T14:02:23.509686Z",
    "modified": "2023-05-31T14:02:23.693911Z",
    "accounts_file": {
      "uri": "http://localhost:8000/v1/portfolios/2/accounts_file/",
      "name": "SourceAccOEDPiWind.csv",
      "stored": "49fc8737b905479089c46b87402f4c10.csv"
    },
    "location_file": {
      "uri": "http://localhost:8000/v1/portfolios/2/location_file/",
      "name": "SourceLocOEDPiWind10.csv",
      "stored": "9a41e6ef038345b6aeab43dddce2ab7d.csv"
    },
    "reinsurance_info_file": None,
    "reinsurance_scope_file": None,
    "storage_links": "http://localhost:8000/v1/portfolios/2/storage_links/",
    "groups": []
  }
]

RETURN_ANALYSIS = [
  {
    "created": "2023-05-31T14:41:22.220161Z",
    "modified": "2023-05-31T14:49:31.126807Z",
    "name": "Analysis_31052023-154122",
    "id": 4,
    "portfolio": 1,
    "model": 1,
    "status": "INPUTS_GENERATION_ERROR",
    "task_started": "2023-05-31T14:41:24.548222Z",
    "task_finished": "2023-05-31T14:49:31.123662Z",
    "complex_model_data_files": [],
    "groups": [],
    "analysis_chunks": None,
    "lookup_chunks": 1,
    "sub_task_count": 7,
    "sub_task_list": "http://localhost:8000/v1/analyses/4/sub_task_list/",
    "sub_task_error_ids": [
      41
    ],
    "status_count": {
      "TOTAL_IN_QUEUE": 5,
      "TOTAL": 7,
      "PENDING": 5,
      "QUEUED": 0,
      "STARTED": 0,
      "COMPLETED": 1,
      "CANCELLED": 0,
      "ERROR": 1
    },
    "input_file": None,
    "settings_file": "http://localhost:8000/v1/analyses/4/settings_file/",
    "settings": "http://localhost:8000/v1/analyses/4/settings/",
    "lookup_errors_file": None,
    "lookup_success_file": None,
    "lookup_validation_file": None,
    "summary_levels_file": None,
    "input_generation_traceback_file": "http://localhost:8000/v1/analyses/4/input_generation_traceback_file/",
    "output_file": None,
    "run_traceback_file": None,
    "run_log_file": None,
    "storage_links": "http://localhost:8000/v1/analyses/4/storage_links/"
  },
  {
    "created": "2023-05-31T14:51:34.565777Z",
    "modified": "2023-05-31T14:55:36.207824Z",
    "name": "Analysis_31052023-155134",
    "id": 6,
    "portfolio": 1,
    "model": 1,
    "status": "READY",
    "task_started": "2023-05-31T14:51:36.849116Z",
    "task_finished": "2023-05-31T14:55:36.192480Z",
    "complex_model_data_files": [],
    "groups": [],
    "analysis_chunks": None,
    "lookup_chunks": 1,
    "sub_task_count": 7,
    "sub_task_list": "http://localhost:8000/v1/analyses/6/sub_task_list/",
    "sub_task_error_ids": [],
    "status_count": {
      "TOTAL_IN_QUEUE": 0,
      "TOTAL": 7,
      "PENDING": 0,
      "QUEUED": 0,
      "STARTED": 0,
      "COMPLETED": 7,
      "CANCELLED": 0,
      "ERROR": 0
    },
    "input_file": "http://localhost:8000/v1/analyses/6/input_file/",
    "settings_file": "http://localhost:8000/v1/analyses/6/settings_file/",
    "settings": "http://localhost:8000/v1/analyses/6/settings/",
    "lookup_errors_file": "http://localhost:8000/v1/analyses/6/lookup_errors_file/",
    "lookup_success_file": "http://localhost:8000/v1/analyses/6/lookup_success_file/",
    "lookup_validation_file": "http://localhost:8000/v1/analyses/6/lookup_validation_file/",
    "summary_levels_file": "http://localhost:8000/v1/analyses/6/summary_levels_file/",
    "input_generation_traceback_file": None,
    "output_file": None,
    "run_traceback_file": None,
    "run_log_file": None,
    "storage_links": "http://localhost:8000/v1/analyses/6/storage_links/"
  },
  {
    "created": "2023-05-31T14:55:35.464693Z",
    "modified": "2023-05-31T14:56:21.522318Z",
    "name": "Analysis_31052023-155535",
    "id": 7,
    "portfolio": 2,
    "model": 2,
    "status": "RUN_COMPLETED",
    "task_started": "2023-05-31T14:55:43.982703Z",
    "task_finished": "2023-05-31T14:56:21.513045Z",
    "complex_model_data_files": [],
    "groups": [],
    "analysis_chunks": 1,
    "lookup_chunks": 1,
    "sub_task_count": 6,
    "sub_task_list": "http://localhost:8000/v1/analyses/7/sub_task_list/",
    "sub_task_error_ids": [],
    "status_count": {
      "TOTAL_IN_QUEUE": 0,
      "TOTAL": 6,
      "PENDING": 0,
      "QUEUED": 0,
      "STARTED": 0,
      "COMPLETED": 6,
      "CANCELLED": 0,
      "ERROR": 0
    },
    "input_file": "http://localhost:8000/v1/analyses/7/input_file/",
    "settings_file": "http://localhost:8000/v1/analyses/7/settings_file/",
    "settings": "http://localhost:8000/v1/analyses/7/settings/",
    "lookup_errors_file": "http://localhost:8000/v1/analyses/7/lookup_errors_file/",
    "lookup_success_file": "http://localhost:8000/v1/analyses/7/lookup_success_file/",
    "lookup_validation_file": "http://localhost:8000/v1/analyses/7/lookup_validation_file/",
    "summary_levels_file": "http://localhost:8000/v1/analyses/7/summary_levels_file/",
    "input_generation_traceback_file": None,
    "output_file": "http://localhost:8000/v1/analyses/7/output_file/",
    "run_traceback_file": None,
    "run_log_file": "http://localhost:8000/v1/analyses/7/run_log_file/",
    "storage_links": "http://localhost:8000/v1/analyses/7/storage_links/"
  }
]



