from importlib import resources
import json
from jsonschema import validate, ValidationError
import os
from pathlib import Path

from oasislmf.computation.base import ComputationStep
from oasislmf.utils.documentation.jsontomd import DefaultJsonToMarkdownGenerator, RDLS_0_2_0_JsonToMarkdownGenerator


class GenerateModelDocumentation(ComputationStep):
    """
    Generates Model Documentation from schema provided in the model config file
    """
    # Command line options
    step_params = [
        {'name': 'doc_out_dir', 'flag': '-o', 'is_path': True, 'pre_exist': False,
         'help': 'Path to the directory in which to generate the Documentation files', 'default': '.'},
        {'name': 'doc_json', 'flag': '-d', 'is_path': True, 'pre_exist': True, 'required': True,
         'help': 'The json file containing model meta-data for documentation'},
        {'name': 'doc_schema_info', 'flag': '-s', 'is_path': True, 'pre_exist': True, 'required': False,
         'help': 'The schema for the model meta-data json'},

    ]
    chained_commands = []

    def validate_doc_schema(self, schema_path, docjson_path):
        """Validates docjson_path file with schema_path file
        Args:
            schema_path (str | os.PathLike): Schema path file
            docjson_path (str | os.PathLike): Documentation JSON path file
        Returns:
            docjson (Dict): Json data loaded as a dictionary
        """
        with open(schema_path, "r") as f:
            schema = json.load(f)

        with open(docjson_path, "r") as f:
            docjson = json.load(f)

        if "datasets" not in docjson:
            raise ValidationError(f"key \'datasets\' not found inside {docjson_path}")

        datasets = docjson["datasets"]
        for i, dataset in enumerate(datasets):
            try:
                validate(instance=dataset, schema=schema)
            except ValidationError as e:
                raise ValidationError(f"doc schema validation error for dataset idx {i}: {e.message}")

        return docjson, schema

    def json_to_mdtxt(self, json_data, full_schema, data_path, doc_out_dir):
        """Convert json data to markdown text with schemas provided
        Args:
            json_data (dict): Json data as dictionary
            full_schema (dict): Full schema file as dictionary
            data_path (str | os.PathLike): Path to data folder for any relative file paths
            doc_out_dir (str | os.PathLike): Path to documentation file output folder for any relative file paths
        """
        schema_id = full_schema["$id"]
        json_to_md_generator = DefaultJsonToMarkdownGenerator
        if schema_id == "https://docs.riskdatalibrary.org/en/0__2__0/rdls_schema.json":
            # RDLS v0.2
            json_to_md_generator = RDLS_0_2_0_JsonToMarkdownGenerator
        else:
            self.logger.warning(f"WARN: Unsupported formatting for following schema: {schema_id}. Using DefaultJsonToMarkdownGenerator output")
        gen = json_to_md_generator(full_schema, data_path, doc_out_dir)
        return gen.generate(json_data, generate_toc=True)

    def run(self):
        if not os.path.exists(self.doc_json):
            raise FileNotFoundError(f'Could not locate doc_json file: {self.doc_json}, Cannot generate documentation')
        if not self.doc_schema_info:
            self.doc_schema_info = resources.files('rdls').joinpath('rdls_schema.json')
            if not os.path.exists(self.doc_schema_info):
                raise FileNotFoundError(f'Could not locate doc_schema_info file: {self.doc_schema_info}, Cannot generate documentation')

        doc_out_dir = Path(self.doc_out_dir)
        doc_json = Path(self.doc_json)
        data_path = doc_json.parent
        doc_schema_info = Path(self.doc_schema_info)
        doc_file = Path(doc_out_dir, 'doc.md')
        json_data, schema = self.validate_doc_schema(doc_schema_info, doc_json)

        with open(doc_file, "w") as f:
            mdtxt = self.json_to_mdtxt(json_data, schema, data_path, doc_out_dir)
            f.write(mdtxt)
