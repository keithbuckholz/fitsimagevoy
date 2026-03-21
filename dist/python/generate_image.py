import sys
from astropy.io import fits
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
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

def generate_file(fits_file, colormap, scale, vmin=None, vmax=None):
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
            # Read in data and header
            data = hdu.data
            header = convert_header_to_dict(hdu.header)

            value_of_xtension = header.get('XTENSION', None)
            if (data is not None)or (i != 0 and value_of_xtension == 'IMAGE'):
                try: 
                    # Set image size
                    divisor = min(header['NAXIS1']/4, header['NAXIS2']/4)
                    plt.figure(figsize=(header['NAXIS1'] / divisor, header['NAXIS2'] / divisor))

                    # Begin gathering image keyword arguments
                    img_kwargs: dict[str, Any] = {"origin": "lower"}

                    # Set colormap
                    if colormap == 'aips0':
                        img_kwargs["cmap"] = plt.get_cmap('nipy_spectral', 8)
                    else:
                        img_kwargs["cmap"] = colormap
                    
                    # Set vmin & vmax
                    if vmin is not None:
                        img_kwargs["vmin"] = vmin
                    if vmax is not None:
                        img_kwargs["vmax"] = vmax

                    # Plot the image with the given scale
                    if scale == 'linear':
                        plt.imshow(data, **img_kwargs)
                    elif scale == 'logarithmic':
                        offset_data = data + np.abs(np.nanmin(data))
                        min_value = np.log(np.min(offset_data[offset_data > 0]))
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
                    selected_options = f"loaded image options: ( {colormap} - {scale} - {clim[0]:.02} - {clim[1]:.02})"
                    plt.title(selected_options)
                    
                    # Make figure background translucent (general theme compatability)
                    fig = plt.gcf()
                    ax = plt.gca()
                    fig.patch.set_alpha(0.0)
                    ax.set_facecolor("none")
                    
                    # Generate the HTML code for the plot
                    html_plot = mpld3.fig_to_html(fig, figid="mpld3Figure2")
                    plt.close()

                    # Add the header and the encoded image to the result
                    result[f'hdu{i}'] = {
                        'header': header,
                        'encoded_image': True,
                        'html_plot': html_plot,
                        'clim': clim,
                    }
                except:
                    # If no image is present in the HDU, only add the header
                    result[f'hdu{i}'] = {
                        'header': header,
                        'encoded_image': False,
                        'html_plot': None,
                        'clim': None,
                    }

            else:
                # If no image is present in the HDU, only add the header
                result[f'hdu{i}'] = {
                    'header': header,
                    'encoded_image': False,
                    'html_plot': None,
                    'clim': None,
                }

    # Return the result as a JSON string
    return json.dumps(result)

if __name__ == "__main__":

    # setup argparser and let it throw errors for missing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("fits_file", type=str)
    parser.add_argument("colormap", type=str)
    parser.add_argument("scale", type=str)
    parser.add_argument("--vmin", type=float, default=None, nargs='?')
    parser.add_argument("--vmax", type=float, default=None, nargs='?')
    args = parser.parse_args()

    fits_file = args.fits_file
    colormap = args.colormap
    scale = args.scale
    clim = {}
    if args.vmin is not None:
        clim["vmin"] = args.vmin
    if args.vmax is not None:
        clim["vmax"] = args.vmax

    try:
        # Call the function and get the encoded image
        file = generate_file(fits_file, colormap, scale, **clim)
        print(file)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)