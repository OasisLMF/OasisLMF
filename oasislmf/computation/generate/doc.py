from collections import defaultdict
import csv
from importlib import resources
import json
from jsonschema import validate, ValidationError
import os
from pathlib import Path
import re

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

    def json_to_mdtxt(self, data, properties_schema, full_schema, md, data_path, header_level):
        """Convert json data to markdown text with schemas provided
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
                    if entry_fmt == "csv":
                        if "download_url" not in entry:
                            md.add_text("No path found to display data")
                            continue
                        fp = Path(data_path, entry["download_url"])
                        if not fp.exists():
                            md.add_text(f"No file found at {str(fp)}, could not display data")
                        else:
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
                    self.json_to_mdtxt(value, properties_schema[key]["properties"], full_schema, md, data_path, header_level + 1)
                else:
                    md.add_header(key_title, level=header_level)
                    md.add_text(value)
            elif "$ref" in properties_schema[key]:
                md.add_header(key_title, level=header_level)
                ref_schema = self._resolve_internal_ref(full_schema, properties_schema[key]["$ref"])
                self.json_to_mdtxt(value, ref_schema["properties"], full_schema, md, data_path, header_level + 1)
            else:
                md.add_header(key_title, level=header_level)
                md.add_text(value)

    def slugify(self, title):
        """Make title strings slugified (transform to URL friendly string)
        Args:
            title (str): Original Title str
        Returns:
            slug_title (str): Slugified Title str
        """
        slug = re.sub(r'[^\w\s-]', '', title).strip().lower()
        slug = re.sub(r'\s+', '-', slug)
        return slug

    def generate_toc(self, markdown_text):
        """Generate a table of contents from markdown string
        Args:
            markdown_text (str): Markdown text
        Returns:
            toc (str): Table of contents markdown string 
        """
        lines = markdown_text.split('\n')
        toc = []
        slug_counts = defaultdict(int)

        for line in lines:
            match = re.match(r'^(#{2,6})\s+(.*)', line)
            if match:
                level = len(match.group(1)) - 1
                title = match.group(2).strip()
                base_slug = self.slugify(title)
                slug_counts[base_slug] += 1
                anchor = base_slug if slug_counts[base_slug] == 1 else f"{base_slug}-{slug_counts[base_slug] - 1}"
                toc.append(f"{'  ' * level}- [{title}](#{anchor})")

        return "## Table of Contents\n\n" + "\n".join(toc)

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
            md = MarkdownGenerator()
            for i, dataset in enumerate(json_data["datasets"]):
                md.add_header(dataset["title"], level=2)
                self.json_to_mdtxt(dataset, schema["properties"], schema, md, data_path, 3)
                md.add_text("")
            mdtxt = md.get_markdown()
            toc = self.generate_toc(mdtxt)
            mdtxt = "# Documentation \n\n" + toc + "\n\n" + mdtxt
            f.write(mdtxt)
