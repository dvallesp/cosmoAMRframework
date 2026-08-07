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


############################################
# Block: Grid hierarchy
############################################

def read_hierarchy_file(hierarchy_file_path):
    """
    Reads the ENZO AMR hierarchy file

    Args:
        hierarchy_file_path (str): path to the .hierarchy file; or to the basic file of the snapshot
            (the parameter file).

    Returns:
        dict: Dictionary indexed by Grid ID. Each entry is another dictionary containing all the 
            properties of that grid.

    Caution! The dictionary will be indexed by the ENZO grid ID, which is generally one more than the 
        grid index that we use in cosmoAMRframework (which is 0-indexed).
    """
    if not hierarchy_file_path.endswith(".hierarchy"):
        hierarchy_file_path = hierarchy_file_path + '.hierarchy'

    grids = {}
    current_grid = None

    with open(hierarchy_file_path, "r") as f:
        for line in f:
            line = line.strip()

            # Skip blank lines
            if not line:
                continue

            # Beginning of a new grid
            if line.startswith("Grid ="):
                grid_id = int(line.split("=")[1])
                current_grid = {}
                grids[grid_id] = current_grid
                continue

            if current_grid is None:
                continue

            # Normal key=value lines
            if "=" in line:
                key, value = line.split("=", 1)
                current_grid[key.strip()] = parse_value(value.strip())

    return grids

def reprocess_grid_hierarchy(hierarchy, Units, ghost_zones=3):
    """ 
    Reprocesses the grid hierarchy and returns the relevant arrays for the simulation, so that these data can be interoperated by the
    cosmoAMRframework.

    Args:
        hierarchy (dict): The grid hierarchy as returned by read_hierarchy_file.
        Units (EnzoUnits): The ENZO units as computed from the parameter file.
        ghost_zones (int): The number of ghost zones in the simulation (default is 3).

    Returns:
        - grids (dict): The reprocessed grid hierarchy with additional information.
        - grids_data (dict): A dictionary containing arrays of grid properties for easier access and interoperability with the cosmoAMRframework.

    Caution! All references to grids here will use the 0-indexed convention, which is different from the 
        ENZO grid ID convention (which is 1-indexed, we call it patch_ID). 
        
        The returned grids_data will have all indices adjusted to be 0-indexed.
        We also keep the patch_ID for redundancy.
    """
    # copy the dictionary ensuring that v is not a reference to the original dictionary
    grids = {k: v.copy() for k, v in hierarchy.items()}

    # find out grid level, patchx,y,z (coordinates of the leftmost real parent cell),
    # patchrx,ry,rz (real coordinates of the centre of that cell), and patchnx,ny,nz (number of real cells in each direction)
    for ig in grids.keys():
        grid = grids[ig]
        
        # Check we specified the correct buffer
        buf = grid['GridStartIndex'][0]
        if buf != ghost_zones:
            print('Warning! The number of ghost zones in the hierarchy file is not equal to the one specified in the function call. Using the one from the hierarchy file.')
            ghost_zones = buf

        xl, yl, zl = grid['GridLeftEdge']
        xr, yr, zr = grid['GridRightEdge']
        nx, ny, nz = np.array(grid['GridDimension'])-ghost_zones*2
        if xr-xl == 1. and yr-yl == 1. and zr-zl == 1.:
            grid['GridLevel'] = 0
            dx = 1./nx
            dy = 1./ny
            dz = 1./nz
            # dx, dy, dz are the base grid ones
        else:
            dxp = (xr-xl)/nx 
            lev = np.log2(dx/dxp)
            # raise an error if lev is not an integer
            if not np.isclose(lev, int(lev)):
                raise ValueError(f"Grid {ig} has a non-integer level: {lev}. Check the grid dimensions and edges.")
            grid['GridLevel'] = int(lev)

    # do another pass and correct the hierarchy file for the wrongly placed pointers! 
    # 1st remove all pointers from grids
    for ig in grids.keys():
        delete_list = []
        for k, v in grids[ig].items():
            if k.startswith('Pointer: Grid['):
                delete_list.append(k)
        for k in delete_list:
            del grids[ig][k]

    # 2nd add the pointers to the correct grids
    p_NextGridThisLevel = {}
    p_NextGridNextLevel = {}
    for ig in grids.keys():
        for k, v in hierarchy[ig].items(): # the original, unmodified one! 
            if k.startswith('Pointer: Grid['):
                # extract the number with a regex 
                match = re.search(r'Pointer: Grid\[(\d+)\]', k)
                #print(match, match.group(1) if match else None)
                pointer_grid_id = int(match.group(1))
                grids[pointer_grid_id][k] = v

                if 'NextGridThisLevel' in k:
                    p_NextGridThisLevel[pointer_grid_id] = v
                elif 'NextGridNextLevel' in k:
                    p_NextGridNextLevel[pointer_grid_id] = v

    # 3rd identify parent grids, sibling grids, child grids,
    for ig in grids.keys():
        children = []
        igchild = p_NextGridNextLevel[ig]

        if igchild > 0:
            children.append(igchild)
            grids[igchild]['parent'] = ig

            while p_NextGridThisLevel[igchild] != 0:
                igchild = p_NextGridThisLevel[igchild]
                grids[igchild]['parent'] = ig
                children.append(igchild)

        grids[ig]['children'] = children

    for ig in grids.keys():
        children = grids[ig]['children']
        if len(children) > 0:
            for child in children:
                if 'siblings' not in grids[child]:
                    grids[child]['siblings'] = []
                grids[child]['siblings'].extend([c for c in children if c != child])
    grids[1]['siblings'] = [] # root grid has no siblings
    
    # 4th identify cousin grids; i.e. cells that, being at the same level, but regardless of their parentage, overlap in space.
    #  in principle, in ENZO, this should not be happening as grids at the same level don't overlap (in the active zone)
    for ig in grids.keys():
        grids[ig]['cousins'] = []

    maxlev = max([grids[ig]['GridLevel'] for ig in grids.keys()])
    for l in range(1, maxlev + 1):
        #print(f"Finding cousins at level {l}...")
        grids_at_level = [ig for ig in grids.keys() if grids[ig]['GridLevel'] == l]
        for ig1 in grids_at_level:
            for ig2 in grids_at_level:
                if ig1 >= ig2:
                    continue 
                #if ig2 in grids[ig1]['siblings']:
                #    continue 
                xl1, yl1, zl1 = grids[ig1]['GridLeftEdge']
                xr1, yr1, zr1 = grids[ig1]['GridRightEdge']
                xl2, yl2, zl2 = grids[ig2]['GridLeftEdge']
                xr2, yr2, zr2 = grids[ig2]['GridRightEdge']
                # check if they overlap in the three directions
                #if l == 2:
                #    print(f"Checking grids {ig1} and {ig2} at level {l}:")
                #    print(f"Grid {ig1}: x=({xl1}, {xr1}), y=({yl1}, {yr1}), z=({zl1}, {zr1})")
                #    print(f"Grid {ig2}: x=({xl2}, {xr2}), y=({yl2}, {yr2}), z=({zl2}, {zr2})")
                if (xl1 < xr2 and xl2 < xr1) and (yl1 < yr2 and yl2 < yr1) and (zl1 < zr2 and zl2 < zr1):
                    grids[ig1]['cousins'].append(ig2)
                    grids[ig2]['cousins'].append(ig1)

    if sum([len(grids[ig]['cousins']) for ig in grids.keys()]) > 0:
        print('\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print("Warning: there are overlapping grids at the same level (cousings). Is this expected?")
        print('\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    grids_data = {} # this is our main new output here

    maxlev = max([grids[ig]['GridLevel'] for ig in grids.keys()])
    grids_data['nlev'] = maxlev
    numpatches = len(grids)
    grids_data['numpatches'] = numpatches

    grids_data['npatch'] = np.zeros(maxlev+1, dtype=int)
    grids_data['patch_ID'] = np.zeros(numpatches, dtype=int)
    grids_data['patchx'] = np.zeros(numpatches, dtype=int)
    grids_data['patchy'] = np.zeros(numpatches, dtype=int)
    grids_data['patchz'] = np.zeros(numpatches, dtype=int)
    grids_data['patchrx'] = np.zeros(numpatches, dtype=np.float32)
    grids_data['patchry'] = np.zeros(numpatches, dtype=np.float32)
    grids_data['patchrz'] = np.zeros(numpatches, dtype=np.float32)
    grids_data['patchnx'] = np.zeros(numpatches, dtype=int)
    grids_data['patchny'] = np.zeros(numpatches, dtype=int)
    grids_data['patchnz'] = np.zeros(numpatches, dtype=int)
    grids_data['pare'] = np.zeros(numpatches, dtype=int)
    grids_data['siblings'] = [None for _ in range(numpatches)]
    grids_data['children'] = [None for _ in range(numpatches)]
    grids_data['cousins'] = [None for _ in range(numpatches)]
    grids_data['levels'] = np.zeros(numpatches, dtype=int)
    grids_data['cpu'] = np.zeros(numpatches, dtype=int)
    grids_data['numpart'] = np.zeros(numpatches, dtype=int)

    for ig in grids.keys():
        iglists = ig - 1

        grid = grids[ig]
        lev = int(grid['GridLevel'])

        dxp = dx / 2**lev
        dyp = dy / 2**lev
        dzp = dz / 2**lev

        nx = grid['GridDimension'][0] - ghost_zones*2
        ny = grid['GridDimension'][1] - ghost_zones*2
        nz = grid['GridDimension'][2] - ghost_zones*2

        xl, yl, zl = grid['GridLeftEdge']
        xr, yr, zr = grid['GridRightEdge']

        assert xl + nx*dxp == xr, f"Grid {ig} has inconsistent dimensions in x: {xl} + {nx}*{dxp} != {xr}"
        assert yl + ny*dyp == yr, f"Grid {ig} has inconsistent dimensions in y: {yl} + {ny}*{dyp} != {yr}"
        assert zl + nz*dzp == zr, f"Grid {ig} has inconsistent dimensions in z: {zl} + {nz}*{dzp} != {zr}"

        if lev == 0:
            px = 0
            py = 0
            pz = 0 
            parent = 0
        else:
            # coordinates of the leftmost cell centre (not the left edge of the grid, but the centre of the first cell)
            x1 = xl + 0.5*dxp
            y1 = yl + 0.5*dyp
            z1 = zl + 0.5*dzp

            parent = grid['parent']
            xlparent, ylparent, zlparent = grids[parent]['GridLeftEdge']
            dxparent = dx / 2**(lev-1)
            dyparent = dy / 2**(lev-1)
            dzparent = dz / 2**(lev-1)
            # already in python format (0 is the first cell)
            px = int((x1 - xlparent) / dxparent)
            py = int((y1 - ylparent) / dyparent)
            pz = int((z1 - zlparent) / dzparent)

        # number of patches at each level
        grids_data['npatch'][lev] += 1

        # ENZO's index of each grid; used for the dictionary of grids, which is indexed by the grid ID
        grids_data['patch_ID'][iglists] = ig

        # Coordinate of the leftmost parent cell in its patch
        grids_data['patchx'][iglists] = px # To Do!
        grids_data['patchy'][iglists] = py # To Do!
        grids_data['patchz'][iglists] = pz # To Do!

        # cell center of the leftmost *parent* cell; this is for compatibility with all the cosmoAMRframework codes, whice
        #  was initially developed for MASCLET
        grids_data['patchrx'][iglists] = xl + dxp
        grids_data['patchry'][iglists] = yl + dyp
        grids_data['patchrz'][iglists] = zl + dzp

        # real number of cells in each direction
        grids_data['patchnx'][iglists] = nx
        grids_data['patchny'][iglists] = ny
        grids_data['patchnz'][iglists] = nz

        # parent grid index; -1 if no parent
        grids_data['pare'][iglists] = parent-1 # here it always refers to the index in the list, not to the ID

        # also do sibling, child lists, and cousing lists
        grids_data['siblings'][iglists] = [s-1 for s in grid['siblings']] # here it always refers to the index in the list, not to the ID
        grids_data['children'][iglists] = [c-1 for c in grid['children']] # here it always refers to the index in the list, not to the ID
        grids_data['cousins'][iglists] = [c-1 for c in grid['cousins']] # here it always refers to the index in the list, not to the ID

        # level
        grids_data['levels'][iglists] = lev

        # cpu index 
        grids_data['cpu'][iglists] = grid['Task']

        # number of particles 
        grids_data['numpart'][iglists] = grid['NumberOfParticles']
        
    # To real coordinates 
    grids_data['patchrx'] *= Units.ulength_Mpc
    grids_data['patchry'] *= Units.ulength_Mpc
    grids_data['patchrz'] *= Units.ulength_Mpc

    # Left and right corners of the patch 
    dx = Units.ulength_Mpc / grids_data['patchnx'][0]
    dy = Units.ulength_Mpc / grids_data['patchny'][0]
    dz = Units.ulength_Mpc / grids_data['patchnz'][0]
    grids_data['patch_xl'] = grids_data['patchrx'] - dx / 2**grids_data['levels']
    grids_data['patch_yl'] = grids_data['patchry'] - dy / 2**grids_data['levels']
    grids_data['patch_zl'] = grids_data['patchrz'] - dz / 2**grids_data['levels']
    grids_data['patch_xr'] = grids_data['patch_xl'] + grids_data['patchnx'] * dx / 2**grids_data['levels']
    grids_data['patch_yr'] = grids_data['patch_yl'] + grids_data['patchny'] * dy / 2**grids_data['levels']
    grids_data['patch_zr'] = grids_data['patch_zl'] + grids_data['patchnz'] * dz / 2**grids_data['levels']

    grids_data['BoxSize'] = Units.ulength_Mpc
    grids_data['Nx'] = grids_data['patchnx'][0]
    grids_data['Ny'] = grids_data['patchny'][0]
    grids_data['Nz'] = grids_data['patchnz'][0]
    grids_data['MaxLevel'] = grids_data['levels'].max()

    return grids, grids_data

