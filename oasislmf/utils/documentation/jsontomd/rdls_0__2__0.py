import csv
import json
import logging
from pathlib import Path

from oasislmf.utils.documentation.jsontomd.base import BaseJsonToMarkdownGenerator

logger = logging.getLogger(__name__)


class RDLS_0_2_0_JsonToMarkdownGenerator(BaseJsonToMarkdownGenerator):
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
                    v = f"{pretty_json}"
                else:
                    v = str(v)
                row.append(v)
            rows.append(row)
        self.md.add_table(headers, rows)

    def generate_ds_overview(self, data, properties_schema, header_level):
        ds_section_properties = [
            "id",
            "title",
            "description",
            "risk_data_type",
            "version",
            "purpose",
            "project",
            "details",
        ]

        self.md.add_header("Dataset Overview", level=header_level)

        present_properties = [p for p in ds_section_properties if p in data]
        for p in present_properties:
            if p == "risk_data_type":
                self.md.add_definition(properties_schema[p]["title"], ",".join(data[p]))
            else:
                self.md.add_definition(properties_schema[p]["title"], data[p])

        return set(ds_section_properties)

    def _gen_ds_hazard(self, data, properties_schema, header_level):
        self.md.add_header(properties_schema["hazard"]["title"], level=header_level)
        # TODO: implement

    def _gen_ds_exposure(self, data, properties_schema, header_level):
        exposure_data = data["exposure"]

        self.md.add_header(properties_schema["exposure"]["title"], level=header_level)
        self.md.add_definition(properties_schema["exposure"]["properties"]["category"]["title"], exposure_data["category"])
        if "taxonomy" in exposure_data:
            self.md.add_definition(properties_schema["exposure"]["properties"]["taxonomy"]["title"], exposure_data["taxonomy"])

        exposure_properties_schema = properties_schema["exposure"]["properties"]
        self.md.add_header(exposure_properties_schema["metrics"]["title"], level=header_level + 1)
        gazetteer_items_ref = exposure_properties_schema["metrics"]["items"]["$ref"]
        self.json_array_to_mdtable(exposure_data["metrics"], gazetteer_items_ref)

    def _gen_ds_vulnerability(self, data, properties_schema, header_level):
        vuln_data = data["vulnerability"]
        self.md.add_header(properties_schema["vulnerability"]["title"], level=header_level)

        vuln_properties_schema = properties_schema["vulnerability"]["properties"]

        self.md.add_header("Hazard info", level=header_level + 1)
        headers = ["Key", "Value"]
        rows = []
        rows.append([vuln_properties_schema["hazard_primary"]["title"], vuln_data["hazard_primary"]])
        if "hazard_secondary" in vuln_data:
            rows.append([vuln_properties_schema["hazard_secondary"]["title"], vuln_data["hazard_secondary"]])
        if "hazard_process_primary" in vuln_data:
            rows.append([vuln_properties_schema["hazard_process_primary"]["title"], vuln_data["hazard_process_primary"]])
        if "hazard_process_secondary" in vuln_data:
            rows.append([vuln_properties_schema["hazard_process_secondary"]["title"], vuln_data["hazard_process_secondary"]])
        if "hazard_analysis_type" in vuln_data:
            rows.append([vuln_properties_schema["hazard_analysis_type"]["title"], vuln_data["hazard_analysis_type"]])
        rows.append([vuln_properties_schema["intensity"]["title"], vuln_data["intensity"]])
        self.md.add_table(headers, rows)

        self.md.add_header("Exposure info", level=header_level + 1)
        self.md.add_definition(vuln_properties_schema["category"]["title"], vuln_data["category"])
        if "taxonomy" in vuln_data:
            self.md.add_definition(vuln_properties_schema["taxonomy"]["title"], vuln_data["taxonomy"])
        cost_items_ref = vuln_properties_schema["cost"]["items"]["$ref"]
        self.json_array_to_mdtable(vuln_data["cost"], cost_items_ref)

        self.md.add_header("Vulnerability Impact info", level=header_level + 1)
        impact_ref = vuln_properties_schema["impact"]["$ref"]
        impact_properties_schema = self._resolve_internal_ref(impact_ref)["properties"]
        impact_data = vuln_data["impact"]
        if "type" in impact_data:
            self.md.add_definition(impact_properties_schema["type"]["title"], impact_data["type"])
        if "metric" in impact_data:
            self.md.add_definition(impact_properties_schema["metric"]["title"], impact_data["metric"])
        if "unit" in impact_data:
            self.md.add_definition(impact_properties_schema["unit"]["title"], impact_data["unit"])
        if "base_data_type" in impact_data:
            self.md.add_definition(impact_properties_schema["base_data_type"]["title"], impact_data["base_data_type"])

        self.md.add_header("Vulnerability Spatial info", level=header_level + 1)
        spatial_ref = vuln_properties_schema["spatial"]["$ref"]
        spatial_properties_schema = self._resolve_internal_ref(spatial_ref)["properties"]
        spatial_data = vuln_data["spatial"]
        self._gen_ds_spatial_data(spatial_data, spatial_properties_schema, header_level)

        self.md.add_header("Vulnerability Functions info", level=header_level + 1)
        functions_data = vuln_data["functions"]
        funcs_properties_schema = vuln_properties_schema["functions"]["properties"]
        for f_id, f_data in functions_data.items():
            self.md.add_header(funcs_properties_schema[f_id]["title"], level=header_level + 2)
            headers = ["Key", "Value"]
            rows = [[funcs_properties_schema[f_id]["properties"][k]["title"], v] for k, v in f_data.items()]
            self.md.add_table(headers, rows)

        if "analysis_details" in vuln_data or "se_category" in vuln_data:
            self.md.add_header("Vulnerability Extra info", level=header_level + 1)
            if "analysis_details" in vuln_data:
                self.md.add_definition(vuln_properties_schema["analysis_details"]["title"], vuln_data["analysis_details"])
            if "se_category" in vuln_data:
                self.md.add_header(vuln_properties_schema["se_category"]["title"], header_level + 2)
                class_ref = vuln_properties_schema["se_category"]["$ref"]
                class_properties_schema = self._resolve_internal_ref(class_ref)["properties"]
                headers = ["Key", "Value"]
                rows = [[class_properties_schema[k]["title"], v] for k, v in vuln_data["se_category"].items()]
                self.md.add_table(headers, rows)

    def _gen_ds_loss(self, data, properties_schema, header_level):
        loss_data = data["loss"]["losses"]

        self.md.add_header(properties_schema["loss"]["title"], level=header_level)
        items_ref = properties_schema["loss"]["properties"]["losses"]["items"]["$ref"]
        ref_properties_schema = self._resolve_internal_ref(items_ref)["properties"]

        cost_items = []
        for loss_item in loss_data:
            cost_items.append(loss_item["cost"])
            loss_item["cost"] = loss_item["cost"]["id"]
            if "impact" in loss_item:
                impact_items_ref = ref_properties_schema["impact"]["$ref"]
                impact_ref_properties_schema = self._resolve_internal_ref(impact_items_ref)["properties"]
                loss_item["impact"] = [impact_ref_properties_schema[k]["title"] + ": " + v for k, v in loss_item["impact"].items()]

        self.json_array_to_mdtable(loss_data, items_ref)

        cost_items_ref = ref_properties_schema["cost"]["$ref"]
        self.md.add_header(ref_properties_schema["cost"]["title"], level=header_level + 1)
        self.json_array_to_mdtable(cost_items, cost_items_ref)

    def generate_ds_risk_data_properties(self, data, properties_schema, header_level):
        ds_section_properties = [
            "hazard",
            "exposure",
            "vulnerability",
            "loss",
        ]
        risk_data_types = data["risk_data_type"]
        present_properties = [p for p in ds_section_properties if p in data]

        missing_properties = set(risk_data_types) - set(present_properties)
        extra_properties = set(present_properties) - set(risk_data_types)
        if len(missing_properties) > 0:
            logger.warning(f"Warning: Missing risk data types from json: {list(missing_properties)}")
        if len(extra_properties) > 0:
            logger.warning(f"Warning: Extra risk data types found in json ({list(extra_properties)}) will also be output")

        if "hazard" in data:
            self._gen_ds_hazard(data, properties_schema, header_level)
        if "exposure" in data:
            self._gen_ds_exposure(data, properties_schema, header_level)
        if "vulnerability" in data:
            self._gen_ds_vulnerability(data, properties_schema, header_level)
        if "loss" in data:
            self._gen_ds_loss(data, properties_schema, header_level)

        return set(ds_section_properties)

    def _gen_ds_spatial_data(self, spatial_data, ref_properties_schema, header_level):
        if "scale" in spatial_data:
            self.md.add_definition(ref_properties_schema["scale"]["title"], spatial_data["scale"])
        if "countries" in spatial_data:
            self.md.add_definition(ref_properties_schema["countries"]["title"], ",".join(spatial_data["countries"]))
        if "centroid" in spatial_data:
            self.md.add_definition(ref_properties_schema["centroid"]["title"], spatial_data["centroid"])
        if "bbox" in spatial_data:
            self.md.add_definition(ref_properties_schema["bbox"]["title"], spatial_data["bbox"])
        if "gazetteer_entries" in spatial_data:
            self.md.add_header(ref_properties_schema["gazetteer_entries"]["title"], level=header_level + 1)
            gazetteer_items_ref = ref_properties_schema["gazetteer_entries"]["items"]["$ref"]
            self.json_array_to_mdtable(spatial_data["gazetteer_entries"], gazetteer_items_ref)
        if "geometry" in spatial_data:
            self.md.add_header(ref_properties_schema["geometry"]["title"], level=header_level + 1)
            geometry_ref_properties_schema = self._resolve_internal_ref(ref_properties_schema["geometry"]["$ref"])["properties"]
            if "type" in geometry_ref_properties_schema:
                self.md.add_definition(geometry_ref_properties_schema["type"]["title"], spatial_data["geometry"]["type"])
            if "coordinates" in geometry_ref_properties_schema:
                self.md.add_collapsible_section(json.dumps(spatial_data["geometry"]["coordinates"], indent=2), "Raw Coordinates")

    def generate_ds_spatial_temporal_properties(self, data, properties_schema, header_level):
        ds_section_properties = [
            "spatial",
            "temporal_resolution",
        ]

        self.md.add_header("Spatial and Temporal Coverage", level=header_level)
        temporal_data = data.get("temporal_resolution", "No temporal information found in json.")
        self.md.add_definition(properties_schema["temporal_resolution"]["title"], temporal_data)

        self.md.add_header("Spatial", level=header_level + 1)
        ref_properties_schema = self._resolve_internal_ref(properties_schema["spatial"]["$ref"])["properties"]
        spatial_data = data["spatial"]
        self._gen_ds_spatial_data(spatial_data, ref_properties_schema, header_level)

        return set(ds_section_properties)

    def generate_ds_resources_properties(self, data, properties_schema, header_level):
        ds_section_properties = [
            "resources",
            "sources",
        ]

        resources_data = data["resources"]

        self.md.add_header("Resources and Sources", level=header_level)
        self.md.add_header(properties_schema["resources"]["title"], level=header_level + 1)
        items_ref = properties_schema["resources"]["items"]["$ref"]
        self.json_array_to_mdtable(resources_data, items_ref)

        for entry in resources_data:
            self.md.add_header(entry.get("title", "Unnamed Resource"), level=header_level + 2)
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

        if "sources" in data:
            self.md.add_header(properties_schema["sources"]["title"], level=header_level + 1)
            sources_data = data["sources"]
            items_ref = properties_schema["sources"]["items"]["$ref"]
            self.json_array_to_mdtable(sources_data, items_ref)

        return set(ds_section_properties)

    def generate_ds_owner_contact_properties(self, data, properties_schema, header_level):
        ds_section_properties = [
            "publisher",
            "creator",
            "contact_point",
            "attributions",
        ]

        self.md.add_header("Ownership and Contacts", level=header_level)

        present_properties = [p for p in ["publisher", "creator", "contact_point"] if p in data]
        for p in present_properties:
            self.md.add_header(properties_schema[p]["title"], level=header_level + 1)
            ref_properties_schema = self._resolve_internal_ref(properties_schema[p]["$ref"])["properties"]
            headers = [ref_properties_schema[k]["title"] for k in data[p].keys()]
            rows = [data[p].values()]
            self.md.add_table(headers, rows)

        if "attributions" in data:
            self.md.add_header(properties_schema["attributions"]["title"], level=header_level + 1)
            attributions_data = data["attributions"]
            for v in attributions_data:
                v["entity"] = list(v["entity"].values())
            items_ref = properties_schema["attributions"]["items"]["$ref"]
            self.json_array_to_mdtable(attributions_data, items_ref)

        return set(ds_section_properties)

    def generate_ds_licensing_links_properties(self, data, properties_schema, header_level):
        ds_section_properties = [
            "license",
            "links",
            "referenced_by",
        ]

        self.md.add_header("Licensing and Links", level=header_level)
        self.md.add_definition(properties_schema["license"]["title"], data["license"])
        links_data = data.get("links", [])
        if links_data:
            self.md.add_header("Links", level=header_level + 1)
            self.md.add_list([x["href"] for x in data["links"]])

        if "referenced_by" in data:
            self.md.add_header(properties_schema["referenced_by"]["title"], level=header_level + 1)
            items_ref = properties_schema["referenced_by"]["items"]["$ref"]
            self.json_array_to_mdtable(data["referenced_by"], items_ref)

        return set(ds_section_properties)

    def generate_dataset(self, data, properties_schema, header_level):
        all_properties = set(self.full_schema["properties"].keys())
        ds_overview_properties = self.generate_ds_overview(data, properties_schema, header_level)
        ds_risk_data_properties = self.generate_ds_risk_data_properties(data, properties_schema, header_level)
        ds_spatial_temporal_properties = self.generate_ds_spatial_temporal_properties(data, properties_schema, header_level)
        ds_resources_properties = self.generate_ds_resources_properties(data, properties_schema, header_level)
        ds_owner_contact_properties = self.generate_ds_owner_contact_properties(data, properties_schema, header_level)
        ds_licensing_links_properties = self.generate_ds_licensing_links_properties(data, properties_schema, header_level)

        missing_properties = all_properties - (
            ds_overview_properties |
            ds_risk_data_properties |
            ds_spatial_temporal_properties |
            ds_resources_properties |
            ds_owner_contact_properties |
            ds_licensing_links_properties
        )

        if len(missing_properties) > 0:
            logger.warning(f"Warning: The following dataset properties have not been implemented for markdown output: {list(missing_properties)}")

    def generate(self, json_data, generate_toc=False):
        self.md.add_header("Documentation", level=1)
        for dataset in json_data.get("datasets", []):
            self.md.add_header(dataset["title"], level=1)
            self.generate_dataset(dataset, self.full_schema["properties"], header_level=2)
            self.md.add_text("")
        return self.md.get_markdown(generate_toc=generate_toc)
