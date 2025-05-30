import json

from oasislmf.utils.documentation.jsontomd import BaseJsonToMarkdownGenerator


class DefaultJsonToMarkdownGenerator(BaseJsonToMarkdownGenerator):
    """
    Default JSON to Markdown Generator class.
    Naively iterates through the dict and outputs with limited formatting.
    """

    def json_array_to_mdtable(self, data, ref):
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
                    v = f"<details><summary>Expand</summary>{pretty_json}</details>"
                else:
                    v = str(v)
                row.append(v)
            rows.append(row)
        self.md.add_table(headers, rows)

    def generate_data(self, data, properties_schema, header_level):
        for key, value in data.items():
            key_title = key
            schema = properties_schema.get(key, {})
            if "type" in schema:
                if schema["type"] == "array":
                    self.md.add_header(key_title, level=header_level)
                    items = schema.get("items", {})
                    if isinstance(items, dict) and "$ref" in items:
                        self.json_array_to_mdtable(value, items["$ref"])
                    else:
                        self.md.add_list(value)
                elif schema["type"] == "object":
                    self.md.add_header(key_title, level=header_level)
                    self.generate_data(value, schema.get("properties", {}), header_level + 1)
                else:
                    self.md.add_header(key_title, level=header_level)
                    self.md.add_text(value)
            elif "$ref" in schema:
                self.md.add_header(key_title, level=header_level)
                ref_schema = self._resolve_internal_ref(schema["$ref"])
                self.generate_data(value, ref_schema.get("properties", {}), header_level + 1)
            else:
                self.md.add_header(key_title, level=header_level)
                if isinstance(value, list):
                    for i, v in enumerate(value):
                        self.md.add_header(f"Item {i}", header_level + 1)
                        self.generate_data(v, properties_schema, header_level + 2)
                elif isinstance(value, dict):
                    for k, v in value.items():
                        self.md.add_header(k, level=header_level + 1)
                        self.generate_data(v, properties_schema, header_level + 2)
                else:
                    self.md.add_text(value)

    def generate(self, json_data, generate_toc=False):
        self.md.add_header("Documentation", level=1)
        self.generate_data(json_data, self.full_schema["properties"], header_level=3)
        return self.md.get_markdown(generate_toc=generate_toc)
