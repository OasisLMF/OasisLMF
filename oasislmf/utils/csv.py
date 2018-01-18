import csv


def get_csv_rows_as_dicts(csv_filepath, csv_field_meta=None):
    """
    Generic method that generates a dictionary for each line in a CSV
    file using the CSV headers as keys.
    """
    with open(csv_filepath) as f:
        for row_dict in csv.DictReader(f):
            if not csv_field_meta:
                yield row_dict
            else:
                yield {
                    k: csv_field_meta[k]['validator'](row_dict[csv_field_meta[k]['csv_header']]) for k in csv_field_meta
                }
