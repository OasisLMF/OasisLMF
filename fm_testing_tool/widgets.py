import qgrid
import io
from IPython.display import display, Javascript, clear_output
import fileupload
import os
from ipywidgets import Dropdown
import ipywidgets as widgets
from pandas.core.frame import DataFrame

def show_df(df=None):
    if not isinstance(df, DataFrame):
        return None
    grid_options = {
        'fullWidthRows': True,
        'syncColumnCellResize': True,
        'forceFitColumns': False,
        'defaultColumnWidth': 150,
        'rowHeight': 28,
        'enableColumnReorder': False,
        'enableTextSelectionOnCells': True,
        'editable': True,
        'autoEdit': True,
        'explicitInitialization': True,
        'maxVisibleRows': 15,
        'minVisibleRows': 8,
        'sortable': True,
        'filterable': True,
        'highlightSelectedCell': False,
        'highlightSelectedRow': True
    }
    qgrid_widget = qgrid.QgridWidget(
        df=df, grid_options=grid_options, show_toolbar=True)
    return qgrid_widget


def file_uploader(upload_dir='examples/uploaded', button_label='Upload .CSV file'):
    _upload_widget = fileupload.FileUploadWidget(label=button_label)
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    def _cb(change):
        decoded = io.StringIO(change['owner'].data.decode('utf-8'))
        filename = change['owner'].filename
        fpath = os.path.join(upload_dir,filename)
        print('Uploaded `{}` ({:.2f} kB)'.format(
            fpath, len(decoded.read()) / 2 **10))
        with io.open(fpath, 'w') as fd:
            fd.write(decoded.getvalue())
            decoded.close()

    _upload_widget.observe(_cb, names='data')
    display(_upload_widget)


def select_source_dir(path_dict, examples_dir=None):
    # Function for the button to select user input
    # Add apply selection to pass by ref 'path_dict'
    def get_user_selection(a): # A default arg is needed here, I am guessing to pass self
        # Displays the current value of source_path
        path_dict['source_dir'] = dropbox.value
        #clear_output()
        display(path_dict['source_dir'])
        set_file_paths(dropbox.value)
        set_file_paths(dropbox.value)
        display(path_dict)

        # Execulte jump to next cell
        display(Javascript("var i = Jupyter.notebook.get_selected_index(); Jupyter.notebook.select(i+1);"))

    def is_valid_example(dir_value):
        file_list = ['location', 'account']
        for f in file_list:
            path = os.path.join(dir_value, f'{f}.csv')
            if not os.path.isfile(path):
                return False
        return True        

    # Function to search and load files with standard names
    def set_file_paths(dir_value):
        file_list = ['location', 'account', 'ri_info', 'ri_scope']
        for f in file_list:
            path = os.path.join(dir_value, f'{f}.csv')
            path_dict[f'{f}_path'] = path if os.path.isfile(path) else None

    # Load default value and build examples list
    e = os.path.abspath(examples_dir)
    option_list = [os.path.join(e, o) for o in os.listdir(e) if os.path.isdir(os.path.join(e,o))]
    option_list.sort()

    # if 'examples_dir' is an example dir load it by default 
    if is_valid_example(e):
        path_dict['source_dir'] = e
    else:    
        path_dict['source_dir'] = os.path.join(e, 'uploaded')
    set_file_paths(path_dict['source_dir'])

    # creation of a widget dropdown object for directory selection
    dropbox = widgets.Dropdown(
        options=option_list, # Object to iterate over
        description='Select:', # User defined
        rows=10, # The number of rows to display when showing the box
        interactive=True, # This makes the box interactive,
    );

    # Button to click
    select_button = widgets.Button(
        description='Set source dir', # User defined
        disabled=False
    )

    # Event Handlers
    select_button.on_click(get_user_selection)

    # display the UI
    ui = widgets.HBox([dropbox,select_button]) # pass an array of widgets to the ui
    display(ui)
    display("Default value: {}".format(path_dict['source_dir']))
    display(path_dict)


