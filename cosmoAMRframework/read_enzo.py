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
from cosmoAMRframework.cosmo_tools import background_density
import numpy as np

############################################
# Block: read parameter files 
############################################

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

############################################
# Block: units
############################################

class EnzoUnits:
    """
    Computes the ENZO code units from the simulation parameter file.
    Initializes the units based on the cosmological parameters and box size provided in the 
        parameter file.

    It is initialized passing the dictionary returned by read_simulation_data() to the 
        constructor.

    Returns: the Units object.
    """

    # ---------- Physical constants ----------
    MPC_TO_M = 3.085678e22
    M_TO_MPC = 1.0 / MPC_TO_M

    MPC_TO_CM = MPC_TO_M * 100.0
    CM_TO_MPC = 1.0 / MPC_TO_CM

    MSUN_TO_KG = 1.98848e30
    KG_TO_MSUN = 1.0 / MSUN_TO_KG

    MSUN_TO_G = MSUN_TO_KG * 1000.0
    G_TO_MSUN = 1.0 / MSUN_TO_G

    G_SI = 6.67430e-11
    G_CGS = 6.67430e-8

    YEAR_TO_S = 365.25 * 24 * 3600.


    def __init__(self, params):

        self.h = params["CosmologyHubbleConstantNow"]
        self.omega_m = params["CosmologyOmegaMatterNow"]
        self.omega_b = self.omega_m - params["CosmologyOmegaDarkMatterNow"]

        self.zini = params["CosmologyInitialRedshift"]

        self.nx = params["TopGridDimensions"][0]
        self.box_size = params["CosmologyComovingBoxSize"] / self.h

        self.fb = (self.omega_m - self.omega_b) / self.omega_m

        self._compute_units()


    def _compute_units(self):

        # ---------- Mass ----------
        mass_box = background_density(self.h, self.omega_m) * self.box_size**3
        self.umass_Msun = mass_box / self.nx**3 # no h-1 factors here
        # in Msun/h
        self.umass_Msun_per_h = self.umass_Msun * self.h

        self.umass_kg = self.umass_Msun * self.MSUN_TO_KG
        self.umass_g = self.umass_Msun * self.MSUN_TO_G

        # ---------- Length ----------
        self.ulength_Mpc = self.box_size
        # in Mpc/h
        self.ulength_Mpc_per_h = self.ulength_Mpc * self.h

        self.ulength_m = self.ulength_Mpc * self.MPC_TO_M
        self.ulength_cm = self.ulength_Mpc * self.MPC_TO_CM

        # ---------- Time ----------
        rho0 = (
            background_density(self.h, self.omega_m)
            * self.MSUN_TO_KG
            / self.MPC_TO_M**3
        )

        self.utime_s = 1.0 / np.sqrt(
            4.0 * np.pi
            * self.G_SI
            * rho0
            * (1.0 + self.zini) ** 3
        )

        self.utime_yr = self.utime_s / self.YEAR_TO_S

        # ---------- Velocity ----------
        length_phys_m = self.ulength_m / (1.0 + self.zini)

        self.uvel_ms = length_phys_m / self.utime_s
        self.uvel_kms = self.uvel_ms / 1000.0
        self.uvel_cms = self.uvel_ms * 100.0


    def __repr__(self):

        return (
            f"Mass unit     : {self.umass_Msun:.6e} Msun\n"
            f"Length unit   : {self.ulength_Mpc:.6e} Mpc\n"
            f"Velocity unit : {self.uvel_kms:.6e} km/s\n"
            f"Time unit     : {self.utime_yr:.6e} yr"
        )