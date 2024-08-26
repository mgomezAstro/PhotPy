# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 17:28:36 2024

@author: M.A. Gómez-Muñoz
@affiliation: ICCUB, Spain
@email: mgomez@icc.ub.edu; mgomez_astro@outlook.com
"""
import numpy as np
import warnings
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy.modeling import models, fitting
from astropy.utils.exceptions import AstropyUserWarning
from photutils.aperture import CircularAperture, aperture_photometry
from photutils.background import (
    Background2D,
    MedianBackground,
    MMMBackground,
    SExtractorBackground,
    MeanBackground,
)
from photutils.detection import IRAFStarFinder
from astroquery.vizier import Vizier

# import matplotlib.pyplot as plt


class PhotPy:
    def __init__(self, input_file: str, ext: int = 0):
        self.input_file = input_file
        self.data = fits.getdata(self.input_file, ext=ext)
        self.hdr = fits.getheader(self.input_file, ext=ext)
        self.data_err = np.zeros_like(self.data)
        self.data_bkg = np.zeros_like(self.data)
        self.sources = None

    def get_background(
        self,
        bkg_type: str = "median_bkg",
        box_size: tuple = (10, 10),
        filter_size: tuple = (3, 3),
    ):
        bkg_options = [
            "sigma_clip",
            "mean_bkg",
            "median_bkg",
            "sextractor_bkg",
            "mmm_bkg",
        ]

        if bkg_type == "sigma_clip":
            _, bkg, bkg_std = sigma_clipped_stats(self.data, sigma=3.0)
            self.data_bkg += bkg
            self.data_err += bkg_std
        elif bkg_type == "mean_bkg":
            sigma_clip = SigmaClip(sigma=3.0)
            bkg_estimator = MeanBackground()
            bkg_2d = Background2D(
                self.data,
                box_size,
                filter_size=filter_size,
                sigma_clip=sigma_clip,
                bkg_estimator=bkg_estimator,
            )
            self.data_bkg += bkg_2d.background
            self.data_err += bkg_2d.background_rms
        elif bkg_type == "median_bkg":
            sigma_clip = SigmaClip(sigma=3.0)
            bkg_estimator = MedianBackground()
            bkg_2d = Background2D(
                self.data,
                box_size,
                filter_size=filter_size,
                sigma_clip=sigma_clip,
                bkg_estimator=bkg_estimator,
            )
            self.data_bkg += bkg_2d.background
            self.data_err += bkg_2d.background_rms
        elif bkg_type == "sextractor_bkg":
            sigma_clip = SigmaClip(sigma=3.0)
            bkg_estimator = SExtractorBackground()
            bkg_2d = Background2D(
                self.data,
                box_size,
                filter_size=filter_size,
                sigma_clip=sigma_clip,
                bkg_estimator=bkg_estimator,
            )
            self.data_bkg += bkg_2d.background
            self.data_err += bkg_2d.background_rms
        elif bkg_type == "mmm_bkg":
            sigma_clip = SigmaClip(sigma=3.0)
            bkg_estimator = MMMBackground()
            bkg_2d = Background2D(
                self.data,
                box_size,
                filter_size=filter_size,
                sigma_clip=sigma_clip,
                bkg_estimator=bkg_estimator,
            )
            self.data_bkg += bkg_2d.background
            self.data_err += bkg_2d.background_rms
        else:
            raise ValueError(f"Background must be one of this: {bkg_options}.")

    def get_field_sources(
        self, threshold: float = 5.0, fwhm: float = 5.5, sigma: float = 3.0
    ):
        _, bkg, bkg_std = sigma_clipped_stats(self.data, sigma=sigma)
        starfinder = IRAFStarFinder(threshold * bkg_std, fwhm)
        self.sources = starfinder(self.data - self.data_bkg)

        # Plotting sources
        # positions = np.transpose(
        #     (self.sources["xcentroid"], self.sources["ycentroid"])
        # )
        # apertures = CircularAperture(positions, r=7.0)
        # plt.imshow(
        #     self.data - self.data_bkg,
        #     origin="lower",
        #     norm="asinh",
        #     cmap="Greys",
        # )
        # apertures.plot(color="red", lw=1.5, alpha=0.5)

    def __cog(
        self, aper_min: float, aper_max: float, n_aper: int, aper_id: int
    ):

        mask_roundness = self.sources["roundness"] <= 0.1
        positions = np.transpose(
            (
                self.sources["xcentroid"][mask_roundness],
                self.sources["ycentroid"][mask_roundness],
            )
        )
        radii = np.linspace(aper_min, aper_max, n_aper)
        apertures = [CircularAperture(positions, r) for r in radii]

        phot = aperture_photometry(self.data - self.data_bkg, apertures)

        mk = np.asarray([])
        rk = (
            np.asarray([radii[i] + radii[i - 1] for i in range(1, len(radii))])
            / 2.0
        )

        xx = []
        yy = []
        for i in range(1, len(apertures)):
            tmp_radius = 0.5 * (radii[i] + radii[i - 1])
            mag_1 = -2.5 * np.log10(phot[f"aperture_sum_{i}"])
            mag_2 = -2.5 * np.log10(phot[f"aperture_sum_{i-1}"])
            diffmags = mag_1 - mag_2
            diffmags[diffmags > 0.0] = np.nan
            yy.append(diffmags.data)
            xx.append([tmp_radius] * len(diffmags))
            mk = np.r_[mk, np.nanmedian(diffmags)]

        # fitting a Moffat function to curve-of-growth
        moffat = models.Moffat1D(amplitude=mk[0], x_0=0.0, fixed={"x_0": True})
        fitter = fitting.LMLSQFitter(calc_uncertainties=True)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The fit may be unsuccessful",
                category=AstropyUserWarning,
            )
            moffat_fit = fitter(moffat, rk, mk)

        # for i in range(len(xx)):
        #     plt.scatter(xx[i], yy[i], c="k")
        # plt.scatter(rk, mk)
        # print(sum(mk[aper_id - 1 : :]))

        # # Plotting the model
        # x_model = np.linspace(min(rk), max(rk), 100)
        # plt.plot(x_model, moffat_fit(x_model), "r")
        # plt.xlabel(r"$(r_{k} + r_{k-1}) / 2$")
        # plt.ylabel(r"$m_{k} - m_{k-1}$")
        # plt.show()

        return moffat_fit, radii[aper_id]

    def get_field_photometry(
        self,
        phot_type: str = "aperture",
        aper_min: float = 1.5,
        aper_max: float = 20,
        n_aper: int = 10,
        aper_id: int = 3,
    ):
        aper_corr_fn, aper = self.__cog(aper_min, aper_max, n_aper, aper_id)
        positions = np.transpose(
            (self.sources["xcentroid"], self.sources["ycentroid"])
        )

        # TODO: implement PSF photometry
        if phot_type == "aperture":
            circular_aperture = CircularAperture(positions, aper)
            phot = aperture_photometry(
                self.data - self.data_bkg,
                circular_aperture,
                error=self.data_err,
            )
            flux = phot["aperture_sum"] / float(self.hdr["EXPTIME"])
            e_flux = phot["aperture_sum_err"] / float(self.hdr["EXPTIME"])
            mag = -2.5 * np.log10(flux)
            e_mag = 0.434 * e_flux / flux
        else:
            raise ValueError(
                f"No recognized phot_tyoe. Your input: {phot_type}"
            )

        x_model = np.linspace(aper, 60, 1000)
        aper_corr = np.trapezoid(aper_corr_fn(x_model), x_model)

        self.sources["mag"] = 25.0 + mag + aper_corr
        self.sources["e_mag"] = e_mag
        self.sources["flux"] = flux
        self.sources["e_flux"] = e_flux

    def pix_to_wcs(self):
        positions = np.transpose(
            (self.sources["xcentroid"], self.sources["ycentroid"])
        )

        wcs = WCS(self.hdr)

        positions_wcs = wcs.all_pix2world(positions, 1)
        ra, dec = np.transpose(positions_wcs)

        self.sources["ra"] = ra
        self.sources["dec"] = dec

    def save(self, make_report: bool = True):
        remove_mask_nans = np.isnan(self.sources["mag"])
        self.sources = self.sources[~remove_mask_nans]
        for col in self.sources.colnames:
            if col not in ["id", "npix", "xcentroid", "ycentroid"]:
                self.sources[col].info.format = ".5f"
            elif col == "xcentroid" or col == "ycentroid":
                self.sources[col].info.format = ".3f"

        self.sources.write(
            self.input_file.replace(".fits", "_sources_ph.csv"),
            format="ascii.csv",
            overwrite=True,
        )

        # TODO: Make a reportlab PDF document
        if make_report:
            pass

    def run_pipe(self):
        self.get_background(bkg_type="median_bkg", box_size=(15, 15))
        self.get_field_sources()
        self.get_field_photometry()
        self.pix_to_wcs()
        self.save()


def calibrate(
    input_table: str,
    vizier_catalog: str,
    band: str,
    constrains: dict,
    coords: tuple | list,
):
    """

    Parameters
    ----------
    input_table : str | Table
        CSV table containing mag, e_mag, ra [degrees] and dec [degrees] columns.
    vizier_catalog : str
        Name of the Vizier catalog es shown in Vizier (e.g., II/246/out).
    band : str
        Name of the filter band as in the original Vizier catalog.
    constrains: dict,
        Dictnioary of constrains (e.g., {"Jmag": ">12"}). Column names must coincide with those in Vizier table.
    coords : tuple | list
        A pair of ra and dec coordinates of your object in degrees. This object will be excluded during the calculation of the zero-point.

    Returns
    -------
    None.

    """

    tab = Table.read(input_table, format="ascii.csv")

    tab_coords = SkyCoord(tab["ra"], tab["dec"], unit="deg")
    my_coords = SkyCoord(coords[0], coords[1], unit="deg")
    tab["sep"] = tab_coords.separation(my_coords).arcsec
    mask_my_obj = tab["sep"] == tab["sep"].min()
    my_obj = tab[mask_my_obj]
    tab = tab[~mask_my_obj].copy()

    tab["_RAJ2000"] = tab["ra"]
    tab["_DEJ2000"] = tab["dec"]
    tab["_RAJ2000"].unit = "deg"
    tab["_DEJ2000"].unit = "deg"

    catalog = Vizier(
        catalog=vizier_catalog, column_filters=constrains
    ).query_region(tab, radius="3s")[0]
    catalog = catalog[~np.isnan(catalog[band])]
    catalog = catalog[~np.isnan(catalog["e_" + band])]
    catalog.write("ref_" + input_table, format="ascii.csv", overwrite=True)
    print("Total ref sources: ", len(catalog))

    c_tab = SkyCoord(tab["ra"], tab["dec"], unit="deg")
    mag_difs = []
    for i in range(len(catalog)):
        c = SkyCoord(catalog["RAJ2000"][i], catalog["DEJ2000"][i], unit="deg")
        tab["sep"] = c_tab.separation(c).arcsec
        mask = tab["sep"] == tab["sep"].min()
        mag_difs.append(catalog[band][i].data - tab["mag"][mask].data)

    ZP = np.nanmedian(mag_difs)
    e_ZP = np.nanmedian(np.abs(mag_difs - ZP))

    print(f"ZP {band}: {ZP:.4f} +- {e_ZP:.4f}")

    tab[band] = tab["mag"] + ZP
    tab[band].info.format = ".4f"
    tab[f"e_{band}"] = np.sqrt(tab["e_mag"] ** 2 + e_ZP**2)
    tab[f"e_{band}"].info.format = ".4f"
    tab["ZP"] = ZP
    tab["ZP"].info.format = ".4f"
    tab["e_ZP"] = e_ZP
    tab["e_ZP"].info.format = ".4f"

    my_obj[band] = my_obj["mag"] + ZP
    my_obj[band].info.format = ".4f"
    my_obj[f"e_{band}"] = np.sqrt(my_obj["e_mag"] ** 2 + e_ZP**2)
    my_obj[f"e_{band}"].info.format = ".4f"

    tab.write(
        input_table.replace(".csv", "_cal.csv"),
        overwrite=True,
        format="ascii.csv",
    )
    my_obj.write(
        input_table.replace(".csv", "_obj_cal.csv"),
        overwrite=True,
        format="ascii.csv",
    )
