import os
import numpy as np
import xarray as xr
import subprocess
import tempfile
import shutil


def run_connectivity(rast):
    cond_rast = rast["habval"]
    perm_rast = rast["perm"]

    rasts = [cond_rast, perm_rast]
    names = ["habval", "perm"]

    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create symlinks to executable and copy necessary files
        os.symlink("/app/connectivity/context_cba", f"{tmpdirname}/context_cba")
        os.symlink("/app/connectivity/libcuda_cba.so", f"{tmpdirname}/libcuda_cba.so")
        os.symlink("/app/connectivity/pet291_19_25_lookup.txt", f"{tmpdirname}/pet291_19_25_lookup.txt")
        os.symlink("/app/connectivity/pet291_19_25_translate.txt", f"{tmpdirname}/pet291_19_25_translate.txt")
        shutil.copy("/app/connectivity/contextCBA.par", f"{tmpdirname}/contextCBA.par")

        for rast, name in zip(rasts, names):
            flt = rast.values.tobytes()
            with open(f"{tmpdirname}/{name}.flt", 'wb') as f:
                f.write(flt)

        nrows, ncols = cond_rast.shape

        # Get coordinates
        x_coords = cond_rast.x.values
        y_coords = cond_rast.y.values

        res = 90

        # Get lower left corner coordinates
        xllcorner = x_coords[0] - res/2
        yllcorner = y_coords[-1] - res/2

        # Create header content
        header_content = (
            f"ncols {ncols}\n"
            f"nrows {nrows}\n"
            f"xllcorner {int(xllcorner)}\n"
            f"yllcorner {int(yllcorner)}\n"
            f"cellsize 90\n"
            f"NODATA_value -9999\n"
            f"byteorder LSBFIRST"
        )

        # Write header file
        for i in names:
            with open(f"{tmpdirname}/{i}.hdr", 'w') as f:
                f.write(header_content)
        
        rast.close()

        # run executable while inside nbas directory using synchronous subprocess
        proc = subprocess.run(
            ["./context_cba", "contextCBA.par"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            cwd=tmpdirname,
            text=True
        )
        
        if proc.returncode != 0:
            raise Exception(f"Process failed with exit code {proc.returncode}: {proc.stderr}")

        conn_rast = read_connectivity_output(f"{tmpdirname}/NHV.flt", nrows, ncols)

    return conn_rast

def read_connectivity_output(file_name, nrows, ncols):
    with open(file_name, 'rb') as f:
        data = f.read()
        data = np.frombuffer(data, dtype=np.float32)
        data = data.reshape(nrows, ncols)
        result = xr.DataArray(data, dims=['y', 'x'])
    return result