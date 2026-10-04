# read /Users/peter/Downloads/20261002/clear_flat.fits and add DATE-OBS and FILTER as i'

import astropy.io.fits as fits

with fits.open("/Users/peter/Downloads/20261002/clear_flat_modified.fits") as hdul:
    # Access the primary header
    primary_header = hdul[0].header

    # Add DATE-OBS and FILTER keywords to the header
    primary_header["DATE-OBS"] = (
        "2026-10-03T00:00:00"  # Example date, replace with actual value
    )
    primary_header["FILTER"] = "i'"  # Example filter, replace with actual value

    # Save the modified FITS file
    hdul.writeto(
        "/Users/peter/Downloads/20261002/clear_flat_modified.fits", overwrite=True
    )
