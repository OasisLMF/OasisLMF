from abc import ABC, abstractmethod

from oasislmf.utils.documentation.markdown import MarkdownGenerator


class BaseJsonToMarkdownGenerator(ABC):
    """
    Base JSON to Markdown Generator class
    """

    def __init__(self, full_schema, data_path, doc_out_dir, markdown_generator=None):
        """
        Args:
            full_schema (Dict): Full schema file as dictionary
            data_path (str | os.PathLike): Path to data folder for any relative file paths
            doc_out_dir (str | os.PathLike): Path to documentation file output folder for any relative file paths
            markdown_generator (MarkdownGenerator, optional): MarkdownGenerator class. Defaults to None.
        """
        self.full_schema = full_schema
        self.data_path = data_path
        self.doc_out_dir = doc_out_dir
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
    def generate(self, json_data, generate_toc=False):
        """Top level function to process entire dict to markdown text
        Args:
            json_data (Dict): Json data as dictionary
            generate_toc (bool, Optional): Generate table of contents bool. Defaults to False.
        Returns:
            markdown_txt (str): Markdown text
        """
        pass
