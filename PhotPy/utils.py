from astropy.table import Table
import aplpy


def plot_sources(catalog, fitsfile, centered_on=None, radius=2.0):
    
    fig = aplpy.FITSFigure(fitsfile, north=True)

    sources = Table.read(catalog, format="ascii.csv")
    for source in sources:
        fig.show_circles(source["ra"], source["dec"], 2.0 / 3600.0, color="red")

    if centered_on is not None:
        fig.recenter(centered_on[0], centered_on[1], radius=radius / 60.)

    fig.show_grayscale(invert=True)
    fig.save(fitsfile.replace(".fits", "_cat.png"))
    fig.close()
    
