import csv
import argparse
import tempfile
import os


def reorder_columns_in_place(file_path, desired_order):
    # Create a temporary file to write reordered content
    with tempfile.NamedTemporaryFile(mode='w', newline='', delete=False) as temp_file:
        with open(file_path, mode='r', newline='') as original_file:
            reader = csv.DictReader(original_file)

            # Validate column presence
            if not all(col in reader.fieldnames for col in desired_order):
                missing = [col for col in desired_order if col not in reader.fieldnames]
                raise ValueError(f"Missing required columns in CSV: {missing}")

            writer = csv.DictWriter(temp_file, fieldnames=desired_order)
            writer.writeheader()
            for row in reader:
                writer.writerow({col: row[col] for col in desired_order})

    # Replace the original file with the reordered one
    os.replace(temp_file.name, file_path)


def main():
    parser = argparse.ArgumentParser(description='Reorder CSV columns in-place.')
    parser.add_argument('file', help='Path to the CSV file to modify')
    parser.add_argument(
        '--order', '-o',
        required=True,
        help='Comma-separated list of desired column order, e.g. "level_id,agg_id,layer_id,profile_id"'
    )
    args = parser.parse_args()

    desired_order = [col.strip() for col in args.order.split(',') if col.strip()]

    try:
        reorder_columns_in_place(args.file, desired_order)
        print(f"Reordered columns successfully in '{args.file}'")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
