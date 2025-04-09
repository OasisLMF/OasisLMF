from importlib import resources
import json
from jsonschema import validate, ValidationError
import os
from pathlib import Path

from oasislmf.computation.base import ComputationStep


class MarkdownGenerator:
    def __init__(self):
        """A simple markdown generator for adding markdown strings
        """
        self.sections = []

    def get_markdown(self):
        """Returns markdown string from joined self.sections
        Returns:
            str: Markdown string
        """
        return "".join(self.sections)

    def add_header(self, title, level=1):
        """Adds header to markdown
        Args:
            title (str): Title string
            level (int): Markdown header level. Defaults to 1.
        """
        self.sections.append(f"{'#' * level} {title}\n")

    def add_definition(self, title, content):
        """Adds definition line to markdown in the following format
        **title**: content
        Args:
            title (str): Name
            content (str): Description
        """
        self.sections.append(f"**{title}**: {content}\n\n")

    def add_table(self, headers, rows):
        """Adds a table to markdown with headers and rows
        Args:
            headers (List[str]): Headers
            rows (List[str]): Rows
        """
        if len(rows) > 0:
            assert len(rows[0]) == len(headers), \
                f"Length of rows ({len(rows[0])}) \
                does not equal length of headers \
                ({len(headers)}) for headers:\n {headers}\n"
        table = "| " + " | ".join(headers) + " |\n"
        table += "|" + "|".join(["---"] * len(headers)) + "|\n"
        for row in rows:
            table += "| " + " | ".join(row) + " |\n"
        self.sections.append(table)
        self.sections.append("\n")

    def add_list(self, items):
        """Adds list to markdown
        Args:
            items (List[str]): List of items
        """
        for item in items:
            self.sections.append(f"- {item}\n")
        self.sections.append("\n")

    def add_text(self, content):
        """Adds text to markdown
        Args:
            content (str): Text content
        """
        self.sections.append(content + "\n\n")


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

    def json_to_mdtxt(self, data, schema, md, header_level):
        for key, value in data.items():
            if key == "title":  # This is for the section title
                continue

            key_title = key  # This is for the key Title name
            if "title" in schema["properties"][key]:
                key_title = schema["properties"][key]["title"]

            if "type" in schema["properties"][key]:
                if schema["properties"][key]["type"] == "array":
                    md.add_header(key_title, level=header_level + 1)
                    arr_items = schema["properties"][key]["items"]
                    if isinstance(arr_items, dict) and "$ref" in arr_items:
                        array_schema = self._resolve_internal_ref(schema, arr_items["$ref"])
                        array_keys = array_schema["properties"].keys()
                        headers = []
                        for k in array_keys:
                            if "title" in array_schema["properties"][k]:
                                headers.append(array_schema["properties"][k]["title"])
                            else:
                                headers.append(k)
                        rows = []
                        for entry in value:
                            row = []
                            for k in array_keys:
                                v = entry.get(k, "")
                                if isinstance(v, list):
                                    v = ", ".join(v)
                                else:
                                    v = str(v)
                                row.append(v)
                            rows.append(row)
                        md.add_table(headers, rows)
                    else:
                        md.add_list(value)
                elif schema["properties"][key]["type"] == "object":
                    # TODO: This is an example of where we need some sort of way to handle nested objects,
                    # where sometimes there are tables hidden inside nested dicts.
                    md.add_header(key_title, level=header_level + 1)
                    for obj_key, obj_val in value.items():
                        md.add_definition(obj_key, obj_val)
                else:
                    md.add_definition(key_title, value)
            else:
                md.add_definition(key_title, value)

    def run(self):
        if not os.path.exists(self.doc_json):
            raise FileNotFoundError(f'Could not locate doc_json file: {self.doc_json}, Cannot generate documentation')
        if not self.doc_schema_info:
            self.doc_schema_info = resources.files('rdls').joinpath('rdls_schema.json')
            if not os.path.exists(self.doc_schema_info):
                raise FileNotFoundError(f'Could not locate doc_schema_info file: {self.doc_schema_info}, Cannot generate documentation')

        doc_out_dir = Path(self.doc_out_dir)
        doc_json = Path(self.doc_json)
        doc_schema_info = Path(self.doc_schema_info)
        doc_file = Path(doc_out_dir, 'doc.md')
        json_data, schema = self.validate_doc_schema(doc_schema_info, doc_json)

        with open(doc_file, "w") as f:
            md = MarkdownGenerator()
            md.add_header("Documentation", level=1)
            for i, dataset in enumerate(json_data["datasets"]):
                md.add_header(dataset["title"], level=2)
                self.json_to_mdtxt(dataset, schema, md, 2)
                md.add_text("")
            f.write(md.get_markdown())
