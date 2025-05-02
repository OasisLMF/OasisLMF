__all__ = [
    'MarkdownGenerator'
]


from collections import defaultdict
import re


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

    def add_collapsible_section(self, text, title="Root"):
        """Adds collapsible section to markdown
        Args:
            text (str): contents of collapsible section
            title (str, optional): Collapsible section title text. Defaults to "Root".
        """
        self.add_text(f"<details><summary>{title}</summary>\n\n```json\n" + text + "\n```\n</details>")

    def add_text(self, content):
        """Adds text to markdown
        Args:
            content (Any): Text content
        """
        self.sections.append(f"{content}\n\n")
