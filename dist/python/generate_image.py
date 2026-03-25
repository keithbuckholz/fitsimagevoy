import sys, os
from astropy.io import fits
from astropy.table import Table
import json
import matplotlib.pyplot as plt
import numpy as np
import mpld3
import argparse
from typing import Any

def convert_header_to_dict(header):
    """Convert a Header object to a dictionary

    Parameters
    ----------
    header : astropy.io.fits.header.Header
        The Header object

    Returns
    -------
    dict
        The dictionary representation of the Header object
    """
    # Convert the Header object to a dictionary
    header_dict = dict(header)
    
    # Ensure that the dictionary is JSON serializable
    for key, value in header_dict.items():
        try:
            json.dumps(value)
        except (TypeError, OverflowError):
            # If the value is not JSON serializable, convert it to a string
            header_dict[key] = str(value)
    
    return header_dict

# Create a colorbar with a custom format function
def format_func(value, string):
    """Format the colorbar ticks

    Parameters
    ----------
    value : float
        The value of the tick
    string : str
        The string representation of the tick

    Returns
    -------
    str
        The formatted string
    """
    # Convert the value to scientific notation with one digit after the decimal point
    formatted_str = "{:.1e}".format(value)
    
    # Remove the leading zero in the exponent
    if 'e+0' in formatted_str:
        formatted_str = formatted_str.replace('e+0', 'e+')
    elif 'e-0' in formatted_str:
        formatted_str = formatted_str.replace('e-0', 'e-')
    
    return formatted_str

def save_fig(fig, save_fn, save_path, dpi=300, **_):
        """
        Short helper function to save figures
        """

        plt.title('')
        plt.rcParams.update({'font.size': 12})
        
        save_full_path = os.path.join(save_path, save_fn)
        fig.savefig(save_full_path, dpi=dpi)

        plt.close(fig)

def generate_image(hdu, colormap="viridis", scale="linear", save_mode: bool = False, **kwargs):
    """Generate a Matplotlib image for an image HDU.
    
    Parameters
    ----------
    hdu : astropy.io.fits.hdu
        HDU from fits file
    colormap : str
        The colormap to use
    scale : str
        The scale to use
    vmin : float
        Optional vmin for plt
    vmax : float
        Optional vmax for plt

    Returns
    -------
    str
        html plot object, ready for json-ifying

    """

    header = hdu.header
    data = hdu.data

    # Set image size
    divisor = min(header['NAXIS1']/4, header['NAXIS2']/4)
    plt.figure(figsize=(header['NAXIS1'] / divisor, header['NAXIS2'] / divisor))

    # Begin gathering image keyword arguments
    img_kwargs: dict[str, Any] = {"origin": "lower"}
    if "vmin" in kwargs:
        if kwargs["vmin"] is not None:
            img_kwargs["vmin"] = kwargs["vmin"]
    if "vmax" in kwargs:
        if kwargs["vmax"] is not None:
            img_kwargs["vmax"] = kwargs["vmax"]


    # Set colormap
    if colormap == 'aips0':
        img_kwargs["cmap"] = plt.get_cmap('nipy_spectral', 8)
    else:
        img_kwargs["cmap"] = colormap

    # Plot the image with the given scale
    if scale == 'linear':
        plt.imshow(data, **img_kwargs)
    elif scale == 'logarithmic':
        offset_data = data + np.abs(np.nanmin(data))
        positive_values = offset_data[offset_data > 0]
        if len(positive_values) > 0:
            min_value = np.log(np.min(positive_values))
        else:
            min_value = 0
        offset_data[offset_data <= 0] = 1e-17
        if "vmin" not in img_kwargs:
            img_kwargs["vmin"] = min_value
        plt.imshow(np.log(offset_data), **img_kwargs)
    elif scale == 'sqrt':
        offset_data = data + np.abs(np.nanmin(data))
        offset_data[offset_data <= 0] = 1e-17
        plt.imshow(np.sqrt(offset_data), **img_kwargs)
    elif scale == 'power':
        offset_data = data + np.abs(np.nanmin(data))
        offset_data[offset_data <= 0] = 1e-17
        plt.imshow(offset_data**1.5, **img_kwargs)
    elif scale == 'squared':
        offset_data = data + np.abs(np.nanmin(data))
        offset_data[offset_data <= 0] = 1e-17
        plt.imshow(offset_data**2, **img_kwargs)

    plt.colorbar(orientation='horizontal', format="{x:.1e}")
    clim = plt.gci().get_clim() # pyright: ignore[reportOptionalMemberAccess]

    # Add tag with image options
    selected_options = f"loaded image options: ( {colormap} - {scale} - {float(clim[0]):.02} - {float(clim[1]):.02})"
    plt.title(selected_options)
    
    # Make figure background translucent (general theme compatability)
    fig = plt.gcf()
    ax = plt.gca()

    if save_mode:
        save_fig(fig, **kwargs)
        fig = None
    else:
        # Make figure background translucent (general theme compatability)
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")

    return fig, clim

def generate_plot_and_table(hdu, x_col: str | None = None, y_col: str | None = None,
                             save_mode: bool = False, **kwargs):
    """Generate a plot from an HDU table or binary table given x & y column names
    
    Parameters
    ----------
    hdu : astropy.io.fits.hdu
        HDU from fits file
    x_col: str
        Name of column to use for x-axis
    y_col: str
        Name of column to use for y-axis

    Returns
    -------
    str
        html plot object, ready for json-ifying

    """
    data = hdu.data
    table = Table(data)

    if len(colnames := table.colnames) == 1:
        y_col = colnames[0]
        y = np.asarray(table[y_col])
        x = np.arange(1, len(y)+1)
        x_col = "Index"
    else:
        if x_col is None:
            x_col = next(c for c in table.colnames if c != y_col)
        if y_col is None:
            y_col = next(c for c in table.colnames if c != x_col)
        x = np.asarray(table[x_col])
        y = np.asarray(table[y_col])

    # lightweight kwargs parsing
    plot_type = kwargs.get("plot_type") or "line"
    point_size = kwargs.get("point_size")
    x_scale = kwargs.get("table_scale_x")
    y_scale = kwargs.get("table_scale_y")

    fig, ax = plt.subplots()

    valid = np.ones_like(x, dtype=bool)
    valid &= np.isfinite(x) & np.isfinite(y)
    valid &= np.logical_not(np.isnan(x)) & np.logical_not(np.isnan(y))

    if x_scale:
        if x_scale == "log":
            valid &= x > 0
        ax.set_xscale(x_scale.lower())
    if y_scale:
        if y_scale == "log":
            valid &= y > 0
        ax.set_yscale(y_scale.lower())

    x = x[valid]
    y = y[valid]

    if plot_type == "scatter":
        if point_size is None:
            ax.plot(x, y,linestyle='', marker="o")
        else:
            ax.plot(x, y,linestyle='', marker="o", markersize=point_size)
    elif plot_type == "step":
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        ax.step(x, y, where='mid')                 
    elif plot_type == "line_scatter":
        ax.plot(x, y, marker="o", markersize=point_size)
    else:
        ax.plot(x, y)

    # Add tag with image options
    selected_options = f"{y_col} vs {x_col}".title()
    ax.set_title(selected_options)

    if save_mode:
        save_fig(fig, **kwargs)
        fig = None
    else:
        # Make figure background translucent (general theme compatability)
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")

    return fig, table.to_pandas().to_dict()


def generate_file(fits_file, **kwargs):
    """Read fits file and return headers and a Matplotlib plot
    
    Parameters
    ----------
    fits_file : str
        Path to the fits file
    colormap : str
        The colormap to use
    scale : str
        The scale to use
    dark_theme: bool
        Is VS Code using a dark theme

    Returns
    -------
    str
        JSON string with the headers and the encoded image

    """
    result = {}
    
    with fits.open(fits_file, ignore_missing_simple=True) as hdul:
        for i, hdu in enumerate(hdul):
            if kwargs['save_mode']:
                if i != kwargs["save_hdu"]:
                    continue
            # Read in data and header
            header = convert_header_to_dict(hdu.header)

            is_image = isinstance(hdu, (fits.ImageHDU, fits.CompImageHDU))
            is_table = isinstance(hdu, (fits.BinTableHDU, fits.TableHDU))

            # Assume nothing exists, overwrite with reality where things exist
            fig = None
            clim = None
            table = None

            if is_image:
                fig, clim = generate_image(hdu, **kwargs)
            elif is_table:
                fig, table = generate_plot_and_table(hdu, **kwargs)
            
            if fig is None:
                html_plot = None
                encoded_image = False
            else:
                html_plot = mpld3.fig_to_html(fig, figid="mpld3Figure2")
                encoded_image = True
                plt.close(fig)

            result[f'hdu{i}'] = {
                'header': header,
                'encoded_image': encoded_image,
                'html_plot': html_plot,
                'table': table,
                'clim': clim
            }

    # Return the result as a JSON string
    return json.dumps(result)

if __name__ == "__main__":

    # setup argparser and let it throw errors for missing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("fits_file", type=str)
    parser.add_argument("--colormap", type=str)
    parser.add_argument("--scale", type=str)
    parser.add_argument("--vmin", type=float)
    parser.add_argument("--vmax", type=float)
    parser.add_argument("--x_col", type=str)
    parser.add_argument("--y_col", type=str)
    parser.add_argument("--plot_type", type=str)
    parser.add_argument("--point_size", type=float)
    parser.add_argument("--table_scale_x", type=str)
    parser.add_argument("--table_scale_y", type=str)
    parser.add_argument("--save_mode", action="store_true")
    parser.add_argument("--save_hdu", type=int)
    parser.add_argument("--save_fn", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--dpi", type=float)
    args = parser.parse_args()

    arg_dict = vars(args)
    fits_file = arg_dict.pop("fits_file")
    
    try:
        # Call the function and get the encoded image
        file = generate_file(fits_file, **arg_dict)
        print(file)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)