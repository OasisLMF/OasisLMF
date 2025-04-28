import csv
import json
from pathlib import Path

from oasislmf.utils.documentation.jsontomd.base import BaseJsonToMarkdownGenerator


class RDLS_0_2_0_JsonToMarkdownGenerator(BaseJsonToMarkdownGenerator):
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

    def generate_dataset(self, data, properties_schema, header_level):
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
                    self.generate_dataset(value, schema.get("properties", {}), header_level + 1)
                else:
                    self.md.add_header(key_title, level=header_level)
                    self.md.add_text(value)
            elif "$ref" in schema:
                self.md.add_header(key_title, level=header_level)
                ref_schema = self._resolve_internal_ref(schema["$ref"])
                self.generate_dataset(value, ref_schema.get("properties", {}), header_level + 1)
            else:
                self.md.add_header(key_title, level=header_level)
                self.md.add_text(value)

    def generate(self, json_data, generate_toc=False):
        self.md.add_header("Documentation", level=1)
        for dataset in json_data.get("datasets", []):
            self.md.add_header(dataset.get("title", "Untitled"), level=2)
            self.generate_dataset(dataset, self.full_schema["properties"], header_level=3)
            self.md.add_text("")
        return self.md.get_markdown(generate_toc=generate_toc)
