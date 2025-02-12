#!/usr/bin/env python3
"""
Download DREAM Challenge training data and convert it to a Stimulus format.

This script demonstrates how to download a CSV file using 
the requests module without storing it on disk, and then 
converts the CSV data into a Polars DataFrame.
"""

from io import StringIO
import requests
import polars as pl

# URLs to download the data
TRAIN_SEQUENCES_URL = "https://zenodo.org/records/10633252/files/train_sequences.txt?download=1"
TRAIN_URL = "https://zenodo.org/records/10633252/files/train.txt?download=1"
DEBUG_LINE_COUNT = 5

# Output file
OUTPUT_FILE = "dream_train.csv"

def download_dream_to_stimulus_format(path: str, *, debug: bool = False) -> pl.DataFrame:
    """Download the DREAM Challenge training CSV data and return it as a Polars DataFrame.

    Returns:
        pl.DataFrame: A DataFrame containing the CSV data.

    Raises:
        requests.HTTPError: If the HTTP request for the CSV fails.
    """

    if debug:
        response = requests.get(path, stream=True)
        response.raise_for_status()  # Raise an error for bad responses

        lines = []
        for i, line in enumerate(response.iter_lines(decode_unicode=True)): #don't think decode unicode is needed, we only have A,C,T,G and numbers
            if i >= DEBUG_LINE_COUNT:
                break
            lines.append(line)

        csv_content = "\n".join(lines)
        csv_io = StringIO(csv_content)
    else:
        response = requests.get(path)
        response.raise_for_status()  # Raise an error for bad responses
        csv_content = response.text
        csv_io = StringIO(csv_content)

    new_columns = ['sequence','expression']
    column_types = {
        'sequence': pl.Utf8,
        'expression': pl.Float64
    }

    # Use Polars to directly read from the file-like object
    df = pl.read_csv(
        csv_io,
        separator="\t",    # or use the appropriate separator
        has_header=False,
        new_columns=new_columns,
        dtypes=column_types,
        n_rows=DEBUG_LINE_COUNT if debug else None,
    )
    return df


if __name__ == "__main__":
    # Example usage: download and print a preview of the DataFrame
    df_train_sequences = download_dream_to_stimulus_format(path=TRAIN_SEQUENCES_URL, debug=False)
    df_train = download_dream_to_stimulus_format(path=TRAIN_URL, debug=False)
    df = pl.concat([df_train_sequences, df_train])
    df.write_csv(OUTPUT_FILE)