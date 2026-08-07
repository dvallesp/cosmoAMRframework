"""
cosmoAMRframework

Provides several functions to read cosmological AMR simulations (Enzo, MASCLET),
and perform different operations on their data.

read_enzo module
Provides the necessary functions for reading Enzo files and loading them in
memory, as well as handling the units and AMR structure and adapting it to the 
cosmoAMRframework internal structure.

Created by David Vallés
"""

import re

def parse_value(value):
    """
    Converts a string to:
    - int if it is an integer
    - float if it is a float (maybe expressed in scientific notation)
    - list of numbers if there are several
    - string if it cannot be converted

    Note: this is not fail-proof, but it has worked so far with all enzo parameter files I have seen...
    """
    
    # split the value by whitespace to handle multiple values
    parts = value.split()

    if len(parts) == 0:
        return None

    def convert(x):
        # first try int, then float, if both fail return the string
        try:
            if re.fullmatch(r"[+-]?\d+", x):
                return int(x)
            return float(x)
        except ValueError:
            return x

    # if there are several elements -> list
    if len(parts) > 1:
        return [convert(x) for x in parts]

    # a single element
    return convert(parts[0])


def read_simulation_data(simulation_file_path):
    """ 
    Reads the ENZO parameter file written by the simulation output, and returns it as a dictionary.
    
    Args:
        simulation_file_path (str): Path to the ENZO output parameter file 
            (e.g. 'RD0015/RedshiftOutput0015')

    Returns:
        dict: A dictionary containing the simulation parameters.
    """
    params = {}

    with open(simulation_file_path, 'r') as f:
        for line in f:
            # Remove leading/trailing whitespace
            line = line.strip()

            # Skip empty lines ##and comments
            if not line: #or line.startswith('#'):
                continue

            # Split the line into key and value
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()

                # Convert the value to the appropriate type
                params[key] = parse_value(value)

    return params