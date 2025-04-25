__all__ = [
    'MarkdownGenerator'
]


from abc import ABC, abstractmethod
from collections import defaultdict
import csv
import json
import logging
from pathlib import Path
import re

logger = logging.getLogger(__name__)


class MarkdownGenerator:
    def __init__(self):
        """A simple markdown generator for adding markdown strings
        """
        self.sections = []

    def get_markdown(self, generate_toc=False):
        """Returns markdown string from joined self.sections
        Args:
            generate_toc (bool): Generate table of contents bool.
        Returns:
            str: Markdown string
        """
        if generate_toc:
            self.generate_toc()
        return "".join(self.sections)

    def _slugify(self, title):
        """Make title strings slugified (transform to URL friendly string)
        Args:
            title (str): Original Title str
        Returns:
            slug_title (str): Slugified Title str
        """
        slug = re.sub(r'[^\w\s-]', '', title).strip().lower()
        slug = re.sub(r'\s+', '-', slug)
        return slug

    def generate_toc(self, ):
        """Generate a table of contents from markdown string
        Returns:
            toc (str): Table of contents markdown string 
        """
        markdown_text = "".join(self.sections)
        lines = markdown_text.split('\n')
        toc = []
        slug_counts = defaultdict(int)

        for line in lines:
            match = re.match(r'^(#{1,6})\s+(.*)', line)
            if match:
                level = len(match.group(1)) - 1
                title = match.group(2).strip()
                base_slug = self._slugify(title)
                slug_counts[base_slug] += 1
                anchor = base_slug if slug_counts[base_slug] == 1 else f"{base_slug}-{slug_counts[base_slug] - 1}"
                toc.append(f"{'  ' * level}- [{title}](#{anchor})")

        self.sections = ["## Table of Contents\n\n" + "\n".join(toc) + "\n\n"] + self.sections

    def add_header(self, title, level=1):
        """Adds header to markdown
        Args:
            title (Any): Title string
            level (int): Markdown header level. Defaults to 1.
        """
        self.sections.append(f"{'#' * level} {title}\n")

    def add_definition(self, title, content):
        """Adds definition line to markdown in the following format
        **title**: content
        Args:
            title (Any): Name
            content (Any): Description
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
            content (Any): Text content
        """
        self.sections.append(f"{content}\n\n")


class BaseJsonToMarkdownGenerator(ABC):
    """
    Base JSON to Markdown Generator class
    """

    def __init__(self, full_schema, data_path, markdown_generator=None):
        """
        Args:
            full_schema (Dict): Full schema file as dictionary
            data_path (str | os.PathLike): Path to data folder for any relative file paths
            markdown_generator (MarkdownGenerator, optional): MarkdownGenerator class. Defaults to None.
        """
        self.full_schema = full_schema
        self.data_path = data_path
        self.md = markdown_generator
        if not markdown_generator:
            self.md = MarkdownGenerator()

    def _resolve_internal_ref(self, ref):
        """Resolves a $ref in the schema (only internal refs supported).
        Args:
            ref (str): Reference string of format #/$<reftitle>/<refname>
        Returns:
            ref_schema (Dict): Data Properties from reference schema as dictionary
        """
        parts = ref.strip("#/").split("/")
        ref_schema = self.full_schema
        for part in parts:
            ref_schema = ref_schema.get(part, {})
        return ref_schema

    @abstractmethod
    def convert_dataset(self, data, properties_schema, header_level):
        """Recursive function to process a dict to markdown text
        Args:
            data (Dict): Json data as dictionary
            properties_schema (Dict): Data Properties from schema as dictionary
            header_level (int): Header level (number of "#"s to add to headers)
        """
        pass

    @abstractmethod
    def convert(self, json_data, generate_toc=False):
        """Top level function to process entire dict to markdown text
        Args:
            json_data (Dict): Json data as dictionary
            generate_toc (bool, Optional): Generate table of contents bool. Defaults to False.
        Returns:
            markdown_txt (str): Markdown text
        """
        pass


class DefaultJsonToMarkdownGenerator(BaseJsonToMarkdownGenerator):
    """
    Default JSON to Markdown Generator class.
    Naively iterates through the dict and outputs with limited formatting.
    """

    def json_to_mdtable(self, data, ref):
        array_schema = self._resolve_internal_ref(ref)
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
        self.md.add_table(headers, rows)

    def convert_dataset(self, data, properties_schema, header_level):
        for key, value in data.items():
            key_title = key
            schema = properties_schema.get(key, {})
            if "type" in schema:
                if schema["type"] == "array":
                    self.md.add_header(key_title, level=header_level)
                    items = schema.get("items", {})
                    if isinstance(items, dict) and "$ref" in items:
                        self.json_to_mdtable(value, items["$ref"])
                    else:
                        self.md.add_list(value)
                elif schema["type"] == "object":
                    self.md.add_header(key_title, level=header_level)
                    self.convert_dataset(value, schema.get("properties", {}), header_level + 1)
                else:
                    self.md.add_header(key_title, level=header_level)
                    self.md.add_text(value)
            elif "$ref" in schema:
                self.md.add_header(key_title, level=header_level)
                ref_schema = self._resolve_internal_ref(schema["$ref"])
                self.convert_dataset(value, ref_schema.get("properties", {}), header_level + 1)
            else:
                self.md.add_header(key_title, level=header_level)
                if isinstance(value, list):
                    for i, v in enumerate(value):
                        self.md.add_header(f"Item {i}", header_level + 1)
                        self.convert_dataset(v, properties_schema, header_level + 2)
                elif isinstance(value, dict):
                    for k, v in value.items():
                        self.md.add_header(k, level=header_level + 1)
                        self.convert_dataset(v, properties_schema, header_level + 2)
                else:
                    self.md.add_text(value)

    def convert(self, json_data, generate_toc=False):
        self.md.add_header("Documentation", level=1)
        self.convert_dataset(json_data, self.full_schema["properties"], header_level=3)
        return self.md.get_markdown(generate_toc=generate_toc)


class RDLS_0_2_0_JsonToMarkdownGenerator(DefaultJsonToMarkdownGenerator):
    def convert_dataset(self, data, properties_schema, header_level):
        for key, value in data.items():
            if key == "title":
                continue

            key_title = properties_schema.get(key, {}).get("title", key)
            schema = properties_schema.get(key, {})

            if key == "resources":
                self.md.add_header(key_title, level=header_level)
                items_schema = schema.get("items", {})
                self.json_to_mdtable(value, items_schema.get("$ref", ""))

                for entry in value:
                    self.md.add_header(entry.get("title", "Unnamed Resource"), level=header_level + 1)
                    fmt = entry.get("format", "unknown")

                    if "download_url" not in entry:
                        self.md.add_text("No path found to display data")
                        continue

                    fp = Path(self.data_path, entry["download_url"])
                    if not fp.exists():
                        self.md.add_text(f"No file found at {fp}, could not display data")
                        continue

                    self.md.add_text(f"File ({fp.name}) found [here]({fp.as_posix()})")

                    if fmt == "csv":
                        with open(fp) as f:
                            reader = csv.DictReader(f)
                            headers = reader.fieldnames or []
                            rows = [[str(row.get(h, "")) for h in headers] for _, row in zip(range(10), reader)]
                        self.md.add_text("First 10 rows displayed only")
                        self.md.add_table(headers, rows)
                    else:
                        self.md.add_text(f"Cannot display preview for {fmt} files")
            elif "type" in schema:
                if schema["type"] == "array":
                    self.md.add_header(key_title, level=header_level)
                    items = schema.get("items", {})
                    if isinstance(items, dict) and "$ref" in items:
                        self.json_to_mdtable(value, items["$ref"])
                    else:
                        self.md.add_list(value)
                elif schema["type"] == "object":
                    self.md.add_header(key_title, level=header_level)
                    self.convert_dataset(value, schema.get("properties", {}), header_level + 1)
                else:
                    self.md.add_header(key_title, level=header_level)
                    self.md.add_text(value)
            elif "$ref" in schema:
                self.md.add_header(key_title, level=header_level)
                ref_schema = self._resolve_internal_ref(schema["$ref"])
                self.convert_dataset(value, ref_schema.get("properties", {}), header_level + 1)
            else:
                self.md.add_header(key_title, level=header_level)
                self.md.add_text(value)

    def convert(self, json_data, generate_toc=False):
        self.md.add_header("Documentation", level=1)
        for dataset in json_data.get("datasets", []):
            self.md.add_header(dataset.get("title", "Untitled"), level=2)
            self.convert_dataset(dataset, self.full_schema["properties"], header_level=3)
            self.md.add_text("")
        return self.md.get_markdown(generate_toc=generate_toc)
