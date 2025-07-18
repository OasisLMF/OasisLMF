import importlib
import pkgutil


TOOL_MODULE_PACKAGE = "oasislmf.pytools.converters.tools"


def get_tools_by_cli(tools_info: dict, cli_command: str) -> list[str]:
    """Return a list of tool names supported by the given CLI command."""
    return [tool for tool, info in tools_info.items() if cli_command in info.get("cli_support", [])]


def build_tool_info():
    """ Generates the tools information dictionary for conversion tools from the files in TOOL_MODULE_PACKAGE

    Returns:
        tool_indo (dict): TOOL_INFO dict
    """
    tool_info = {}

    package = importlib.import_module(TOOL_MODULE_PACKAGE)
    for _, module_name, _ in pkgutil.iter_modules(package.__path__):
        mod = importlib.import_module(f"{TOOL_MODULE_PACKAGE}.{module_name}")

        tool_info[module_name] = {
            "headers": getattr(mod, "headers", None),
            "dtype": getattr(mod, "dtype", None),
            "fmt": getattr(mod, "fmt", None),
            "cli_support": getattr(mod, "cli_support", []),
        }

    return tool_info


TOOL_INFO = build_tool_info()
SUPPORTED_BINTOCSV = get_tools_by_cli(TOOL_INFO, "bintocsv")
SUPPORTED_CSVTOBIN = get_tools_by_cli(TOOL_INFO, "csvtobin")
SUPPORTED_BINTOPARQUET = get_tools_by_cli(TOOL_INFO, "bintoparquet")
SUPPORTED_PARQUETTOBIN = get_tools_by_cli(TOOL_INFO, "parquettobin")
