from importlib import resources
import json
from jsonschema import validate, ValidationError
import os
from pathlib import Path
from oasislmf.computation.base import ComputationStep
from oasislmf.utils.data import get_utctimestamp


class GenerateModelDocumentation(ComputationStep):
    """
    Generates Model Documentation from schema provided in the model config file
    """
    # Command line options
    step_params = [
        {'name': 'doc_out_dir', 'flag': '-o', 'is_path': True, 'pre_exist': False,
         'help': 'Path to the directory in which to generate the Documentation files'},
        {'name': 'doc_json', 'flag': '-d', 'is_path': True, 'pre_exist': True, 'required': True,
         'help': 'The json file containing model meta-data for documentation'},
        {'name': 'doc_schema_info', 'flag': '-s', 'is_path': True, 'pre_exist': True, 'required': False,
         'help': 'The schema for the model meta-data json'},

    ]
    chained_commands = []

    def _get_output_dir(self):
        if self.doc_out_dir:
            return self.doc_out_dir
        utcnow = get_utctimestamp(fmt='%Y%m%d%H%M%S')
        return os.path.join(os.getcwd(), 'docs', 'files-{}'.format(utcnow))

    def validate_doc_schema(self, schema_path, docjson_path):
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

    def json_to_markdown(self, json_path, md_path):
        # TODO: replace this temporary function with new mdutils builder class
        with open(json_path, "r") as f:
            data = json.load(f)

        lines = ["# JSON Data\n"]

        def render_dict(d, level=2):
            for key, value in d.items():
                header = f"{'#' * level} {key}"
                lines.append(header)

                if isinstance(value, dict):
                    render_dict(value, level + 1)
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        lines.append(f"- **Item {i + 1}**")
                        if isinstance(item, dict):
                            render_dict(item, level + 2)
                        else:
                            lines.append(f"  - {item}")
                else:
                    lines.append(f"**Value:** `{value}`")

        if isinstance(data, dict):
            render_dict(data)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                lines.append(f"## Item {i + 1}")
                if isinstance(item, dict):
                    render_dict(item, level=3)
                else:
                    lines.append(f"`{item}`")
        else:
            lines.append(f"`{data}`")

        with open(md_path, "w") as f:
            f.write("\n\n".join(lines))

    def run(self):
        if not os.path.exists(self.doc_json):
            raise FileNotFoundError(f'Could not locate doc_json file: {self.doc_json}, Cannot generate documentation')
        if not self.doc_schema_info:
            self.doc_schema_info = resources.files('rdls').joinpath('rdls_schema.json')
            if not os.path.exists(self.doc_schema_info):
                raise FileNotFoundError(f'Could not locate doc_schema_info file: {self.doc_schema_info}, Cannot generate documentation')

        doc_out_dir = Path(self._get_output_dir())
        doc_json = Path(self.doc_json)
        doc_schema_info = Path(self.doc_schema_info)
        doc_file = Path(doc_out_dir, 'doc.md')
        self.validate_doc_schema(doc_schema_info, doc_json)

        self.json_to_markdown(doc_json, doc_file)
