import csv
from importlib import resources
import json
from jsonschema import validate, ValidationError
import os
from pathlib import Path

from oasislmf.computation.base import ComputationStep
from oasislmf.utils.mdutils import MarkdownGenerator


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

    def _resolve_internal_ref(self, schema, ref):
        """Resolves a $ref in the schema (only internal refs supported)."""
        parts = ref.lstrip('#/').split('/')
        ref_schema = schema
        for part in parts:
            ref_schema = ref_schema.get(part)
        return ref_schema

    def json_to_mdtable(self, data, full_schema, md, ref):
        array_schema = self._resolve_internal_ref(full_schema, ref)
        array_keys = array_schema["properties"].keys()
        headers = []
        for k in array_keys:
            if "title" in array_schema["properties"][k]:
                headers.append(array_schema["properties"][k]["title"])
            else:
                headers.append(k)
        rows = []
        for entry in data:
            row = []
            for k in array_keys:
                v = entry.get(k, "")
                if isinstance(v, list):
                    rets = []
                    for v_ in v:
                        if isinstance(v_, dict):
                            v_ = json.dumps(v_, indent=4).replace('\n', '<br>').replace(' ', '&nbsp;')
                        rets.append(str(v_))
                    v = ",<br>".join(rets)
                elif isinstance(v, dict):
                    pretty_json = json.dumps(v, indent=4).replace('\n', '<br>').replace(' ', '&nbsp;')
                    v = f"{pretty_json}"
                else:
                    v = str(v)
                row.append(v)
            rows.append(row)
        md.add_table(headers, rows)

    def json_to_mdtxt_default(self, data, properties_schema, full_schema, md, data_path, header_level):
        """Convert json data to markdown text with schemas provided, default formatting, just outputs everything
        Args:
            data (dict): Json data as dictionary
            properties_schema (dict): Data Properties from schema as dictionary
            full_schema (dict): Full schema file as dictionary
            md (MarkdownGenerator): MarkdownGenerator class
            data_path (str | os.PathLike): Path to data folder for any relative file paths
            header_level (int): Header level (number of "#"s to add to headers)
        """
        for key, value in data.items():
            key_title = key
            if "type" in properties_schema[key]:
                if properties_schema[key]["type"] == "array":
                    md.add_header(key_title, level=header_level)
                    arr_items = properties_schema[key]["items"]
                    if isinstance(arr_items, dict) and "$ref" in arr_items:
                        self.json_to_mdtable(value, full_schema, md, arr_items["$ref"])
                    else:
                        md.add_list(value)
                elif properties_schema[key]["type"] == "object":
                    md.add_header(key_title, level=header_level)
                    self.json_to_mdtxt_default(value, properties_schema[key]["properties"], full_schema, md, data_path, header_level + 1)
                else:
                    md.add_header(key_title, level=header_level)
                    md.add_text(value)
            elif "$ref" in properties_schema[key]:
                md.add_header(key_title, level=header_level)
                ref_schema = self._resolve_internal_ref(full_schema, properties_schema[key]["$ref"])
                self.json_to_mdtxt_default(value, ref_schema["properties"], full_schema, md, data_path, header_level + 1)
            else:
                md.add_header(key_title, level=header_level)
                md.add_text(value)

    def json_to_mdtxt_rdls_0__2__0(self, data, properties_schema, full_schema, md, data_path, header_level):
        """Convert json data to markdown text with schemas provided, formatting for rdls v0.2
        Args:
            data (dict): Json data as dictionary
            properties_schema (dict): Data Properties from schema as dictionary
            full_schema (dict): Full schema file as dictionary
            md (MarkdownGenerator): MarkdownGenerator class
            data_path (str | os.PathLike): Path to data folder for any relative file paths
            header_level (int): Header level (number of "#"s to add to headers)
        """
        for key, value in data.items():
            if key == "title":  # This is for the section title
                continue

            key_title = key  # This is for the key Title name
            if "title" in properties_schema[key]:
                key_title = properties_schema[key]["title"]

            if key == "resources":
                md.add_header(key_title, level=header_level)
                arr_items = properties_schema[key]["items"]
                self.json_to_mdtable(value, full_schema, md, arr_items["$ref"])
                for entry in value:
                    md.add_header(entry["title"], level=header_level + 1)
                    entry_fmt = entry["format"]

                    if "download_url" not in entry:
                        md.add_text("No path found to display data")
                        continue

                    fp = Path(data_path, entry["download_url"])
                    if not fp.exists():
                        md.add_text(f"No file found at {str(fp)}, could not display data")
                        continue
                    md.add_text(f"File ({fp.name}) found [here]({fp.as_posix()})")

                    if entry_fmt == "csv":
                        max_rows = 10
                        md.add_text(f"First {max_rows} rows displayed only")
                        with open(fp) as f:
                            reader = csv.DictReader(f)
                            headers = reader.fieldnames or []
                            rows = []

                            for i, row_dict in enumerate(reader):
                                if i >= max_rows:
                                    break
                                # Convert row dict to list in header order
                                row = [str(row_dict.get(h, "")) for h in headers]
                                rows.append(row)
                        md.add_table(headers, rows)
                    else:
                        md.add_text(f"Cannot display preview for {entry_fmt} files")
            # Markdown code for general nonspecific json data
            elif "type" in properties_schema[key]:
                if properties_schema[key]["type"] == "array":
                    md.add_header(key_title, level=header_level)
                    arr_items = properties_schema[key]["items"]
                    if isinstance(arr_items, dict) and "$ref" in arr_items:
                        self.json_to_mdtable(value, full_schema, md, arr_items["$ref"])
                    else:
                        md.add_list(value)
                elif properties_schema[key]["type"] == "object":
                    md.add_header(key_title, level=header_level)
                    self.json_to_mdtxt_rdls_0__2__0(value, properties_schema[key]["properties"], full_schema, md, data_path, header_level + 1)
                else:
                    md.add_header(key_title, level=header_level)
                    md.add_text(value)
            elif "$ref" in properties_schema[key]:
                md.add_header(key_title, level=header_level)
                ref_schema = self._resolve_internal_ref(full_schema, properties_schema[key]["$ref"])
                self.json_to_mdtxt_rdls_0__2__0(value, ref_schema["properties"], full_schema, md, data_path, header_level + 1)
            else:
                md.add_header(key_title, level=header_level)
                md.add_text(value)

    def json_to_mdtxt(self, json_data, full_schema, data_path):
        """Convert json data to markdown text with schemas provided
        Args:
            json_data (dict): Json data as dictionary
            full_schema (dict): Full schema file as dictionary
            data_path (str | os.PathLike): Path to data folder for any relative file paths
        """
        schema_id = full_schema["$id"]
        json_to_mdtxt_func = self.json_to_mdtxt_default
        if schema_id == "https://docs.riskdatalibrary.org/en/0__2__0/rdls_schema.json":
            # RDLS v0.2
            json_to_mdtxt_func = self.json_to_mdtxt_rdls_0__2__0
        else:
            self.logger.warning(f"WARN: Unsupported formatting for following schema: {schema_id}. Using default formatting output")

        md = MarkdownGenerator()
        md.add_header("Documentation", level=1)
        for i, dataset in enumerate(json_data["datasets"]):
            md.add_header(dataset["title"], level=2)
            json_to_mdtxt_func(dataset, full_schema["properties"], full_schema, md, data_path, header_level=3)
            md.add_text("")
        return md.get_markdown(generate_toc=True)

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
            mdtxt = self.json_to_mdtxt(json_data, schema, data_path)
            f.write(mdtxt)
