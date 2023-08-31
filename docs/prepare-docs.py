

from glob import glob
from importlib import import_module

cli_module_names = (m.split('/')[-1].split('.')[0] for m in glob("oasislmf/cli/*") if m.endswith('.py'))

for cli_module_name in cli_module_names:

    main_class = getattr(
        import_module(f"oasislmf.cli.{cli_module_name}"),
        f'{cli_module_name.capitalize()}Cmd',
        None
    )

    print(main_class, cli_module_name)
    if main_class:
        sub_commands = main_class.sub_commands
        print(cli_module_name, sub_commands)
        for command, class_name in sub_commands.items():

            tmpl = f"""
            ``{cli_module_name}``
            {'-'*len(cli_module_name)}
            .. automodule:: oasislmf.cli.{cli_module_name}.{class_name}
            :noindex:
            """

            print(tmpl)
