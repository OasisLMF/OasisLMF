To configure pulling model data from an external source such as S3 or ABFS you need to provide
a model storage configuration file. The path to this file is passed to the model commands using the 
`model_storage_json` flag. This file should take the following form::

    {
        "storage_class": <python path to storage class>
        "options": {
            <...options to pass to the storage class>
        }
    }

For example, to connect to an S3 bucket your configuration may look like::

    {
        "storage_class": "lot3.filestore.backends.aws_s3.AwsS3Storage", 
        "options": {
            "bucket_name": "modeldata", 
            "access_key": "<access_key>",
            "secret_key": "<secret_key>",
            "root_dir": "OasisLMF/PiWind/3"
        }
    }

The root directory provided should point to the root directory where all model files are 
stored.
