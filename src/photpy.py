# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 17:28:36 2024

@author: M.A. Gómez-Muñoz
@affiliation: ICCUB, Spain
@email: mgomez@icc.ub.edu; mgomez_astro@outlook.com
"""
import numpy as np
from scipy.stats import median_abs_deviation
from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy.visualization import simple_norm
from photutils.aperture import (
    CircularAperture,
    CircularAnnulus,
    ApertureStats,
)
from photutils.background import (
    Background2D,
    MedianBackground,
    MMMBackground,
    SExtractorBackground,
    MeanBackground,
)
from photutils.detection import DAOStarFinder
from photutils.profiles import RadialProfile
from astroquery.vizier import Vizier

import matplotlib.pyplot as plt


def get_fwhm(data: np.ndarray, xycen: list[float]) -> float:
    radii = np.arange(20)
    rp = RadialProfile(data, xycen, radii)
    _ = rp.gaussian_fit
    fwhm = rp.gaussian_fwhm

    return fwhm


class PhotPy:
    def __init__(
        self,
        input_file: str,
        ext: int = 0,
    ):
        self.input_file = input_file
        self.data = fits.getdata(self.input_file, ext=ext)
        self.hdr = fits.getheader(self.input_file, ext=ext)
        self.data_err = np.zeros_like(self.data)
        self.data_bkg = np.zeros_like(self.data)
        self.sources: None | Table = None

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
        self,
        threshold: float = 5.0,
        fwhm: float = 5.5,
        **kwargs,
    ):
        _, _, std = sigma_clipped_stats(
            self.data - self.data_bkg, sigma=3.0
        )
        starfinder = DAOStarFinder(
            threshold = threshold * std,
            fwhm = fwhm,
            **kwargs,
        )
        self.sources = starfinder(self.data - self.data_bkg)
        if self.sources is None:
            raise ValueError(
                "Did not found any source in the field. Try another get_field_sources configuration."
            )

        print(f"Total sources found: {len(self.sources)}.")
        if "DATE-OBS" in self.hdr.keys():
            self.sources["mjd"] = Time(self.hdr["DATE-OBS"]).mjd
        elif "DATE_OBS" in self.hdr.keys():
            self.sources["mjd"] = Time(self.hdr["DATE_OBS"]).mjd
        elif "MJD-OBS" in self.hdr.keys():
            self.sources["mjd"] = float(self.hdr["MJD-OBS"])
        else:
            raise ValueError("Not DATE-OBS or DATE_OBS or MJD-OBS found in header.")

    def _get_fwhm(self):
        bkg_fitter = SExtractorBackground()
        bkg = Background2D(self.data, (64, 64), bkg_estimator=bkg_fitter)
        data_bkgsub = self.data - bkg.background
        
        all_fwhm = []
        for source in self.sources[:15]:
            xycen = [source["xcentroid"], source["ycentroid"]]
            all_fwhm.append(get_fwhm(data_bkgsub, xycen))

        _, fwhm_median, _ = sigma_clipped_stats(all_fwhm, sigma=2.0)

        return fwhm_median


    def get_aperture_photometry(self, aper: float = None, annulus: list = None):
        fwhm = self._get_fwhm()
        print(f"Detection FWHM: {fwhm:.2f}")
        positions = np.transpose((self.sources["xcentroid"], self.sources["ycentroid"]))
        aper_pix = CircularAperture(positions, fwhm * 2.5)
        aper_annulus_pix = CircularAnnulus(positions, fwhm * 3.0, fwhm * 4.0)

        if aper is not None:
            print("Setting user aperture instead of 2.5*FWHM.")
            aper_pix = CircularAperture(positions, aper)
            aper_annulus_pix = CircularAnnulus(positions, annulus[0], annulus[1])

        sigclip = SigmaClip(sigma=3.0, maxiters=10)
        aper_stats = ApertureStats(self.data - self.data_bkg, aper_pix, sigma_clip=None)
        bkg_stats = ApertureStats(self.data - self.data_bkg, aper_annulus_pix, sigma_clip=sigclip)

        total_bkg = bkg_stats.median * aper_stats.sum_aper_area.value
        aper_sum_bksub = aper_stats.sum - total_bkg

        gain = 1.0
        if "GAIN" in self.hdr:
            gain = float(self.hdr["GAIN"])
        flux = gain * aper_sum_bksub / float(self.hdr["EXPTIME"])
        mag = -2.5 * np.log10(flux)

        e_flux = np.sqrt(flux + aper_stats.sum_aper_area.value * bkg_stats.mad_std**2)
        # e_mag = np.abs(
        #     -2.5 * np.log10((aper_sum_bksub + e_flux) / self.hdr["EXPTIME"]) - mag
        # )
        e_mag = np.abs(
            mag
            - (-2.5 * np.log10(gain * (aper_sum_bksub + e_flux) / self.hdr["EXPTIME"]))
        )

        self.sources["aper"] = aper_pix.r
        self.sources["r_in"] = aper_annulus_pix.r_in
        self.sources["r_out"] = aper_annulus_pix.r_out
        self.sources["mag"] = mag
        self.sources["e_mag"] = e_mag
        self.sources["flux"] = flux
        self.sources["e_flux"] = e_flux
        self.sources["bkg"] = total_bkg

    def pix_to_wcs(self):
        positions = np.transpose((self.sources["xcentroid"], self.sources["ycentroid"]))

        wcs = WCS(self.hdr)

        positions_wcs = wcs.all_pix2world(positions, 1)
        ra, dec = np.transpose(positions_wcs)

        self.sources["ra"] = ra
        self.sources["dec"] = dec

    def save(self):
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

    def run_pipe(self, remove_background: bool = False):
        if remove_background:
            self.get_background()
        self.get_field_sources()
        self.get_aperture_photometry()
        self.pix_to_wcs()
        self.save()


def calibrate(
    input_table: str,
    vizier_catalog: str,
    band: str,
    constrains: dict,
    coords: tuple | list,
    use_pol_fit: bool = False,
    obs_limits: dict | None = None,
    # color_term : None | str = None,
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

    copy_tab = tab.copy()
    if obs_limits is not None:
        for key in obs_limits.keys():
            mask = eval(f"copy_tab['{key}'] {obs_limits[key]}")
            copy_tab = copy_tab[mask]
        if len(copy_tab) <= 1:
            print("There are no sources that satisfy your obs limits criteria.")
            raise ValueError("No sources found in your data.")

    catalog = Vizier(
        catalog=vizier_catalog,
        column_filters=constrains,
        columns=["all", "_RAJ2000", "_DEJ2000"],
    ).query_region(copy_tab, radius="3s")[0]
    catalog = catalog[~np.isnan(catalog[band])]
    catalog = catalog[~np.isnan(catalog["e_" + band])]
    catalog.write("ref_" + input_table, format="ascii.csv", overwrite=True)
    print("Total ref sources: ", len(catalog))

    c_tab = SkyCoord(tab["ra"], tab["dec"], unit="deg")
    mag_difs = []
    catalog["mag"] = -99.0
    catalog["e_mag"] = -99.0
    for i in range(len(catalog)):
        c = SkyCoord(catalog["_RAJ2000"][i], catalog["_DEJ2000"][i], unit="deg")
        tab["sep"] = c_tab.separation(c).arcsec
        mask = tab["sep"] == tab["sep"].min()
        catalog["mag"][i] = tab["mag"][mask]
        catalog["e_mag"][i] = tab["e_mag"][mask]
        mag_difs.append(catalog[band][i].data - tab["mag"][mask].data)

    mag_difs = np.asarray(mag_difs)
    _, median_ZP, std_ZP = sigma_clipped_stats(mag_difs, sigma=3., maxiters=10, stdfunc="mad_std")
    mask = np.abs(mag_difs - median_ZP) < (3.0 * std_ZP)
    ZP = np.nanmedian(mag_difs[mask]) 
    e_ZP = median_abs_deviation(mag_difs[mask], nan_policy="omit")

    print(f"ZP {band}: {ZP:.4f} +- {e_ZP:.4f}")

    # if color_term is not None:
    #     print(f"Calculating the color term using: {band} - {color_term}")
    #     x_axis_fit = catalog[band] - catalog[color_term]
    #     y_axis_fit = catalog[band] - catalog["mag"] - ZP
    #     _, median_y, std_y = sigma_clipped_stats(y_axis_fit, stdfunc="mad_std")
    #     mask = np.abs(y_axis_fit - median_y) < (3.0 * std_y)
    #     mask = mask * np.abs(x_axis_fit) < 0.8
    #     weights = 1 / np.sqrt(catalog["e_" + band] ** 2 + catalog["e_" + color_term] ** 2)
    #     coeffs = np.polyfit(x_axis_fit[mask], y_axis_fit[mask], w = weights[mask], deg=1)
    #     pol = np.poly1d(coeffs)
    #     plt.scatter(x_axis_fit[mask], y_axis_fit[mask])
    #     plt.plot(x_axis_fit[mask], pol(x_axis_fit[mask]))
    #     plt.show()
    #
    #     _, ZP, e_ZP = sigma_clipped_stats(ZP - (coeffs[0] * x_axis_fit[mask] + coeffs[1]), stdfunc="mad_std")
    #
    #     print(f"ZP {band} (color-corrected): {ZP:.4f} +- {e_ZP:.4f}")

    pol = lambda x: x
    if use_pol_fit:
        x_fit = catalog["mag"] + ZP
        y_fit = catalog[band]
        weights = 1 / catalog["e_" + band]
        coeffs = np.polyfit(x_fit, y_fit, w=weights, deg=1)
        pol = np.poly1d(coeffs)

    tab["m_" + band] = pol(tab["mag"] + ZP)
    tab["m_" + band].info.format = ".4f"
    tab[f"e_m_{band}"] = np.sqrt(tab["e_mag"] ** 2 + e_ZP**2)
    tab[f"e_m_{band}"].info.format = ".4f"
    tab["ZP"] = ZP
    tab["ZP"].info.format = ".4f"
    tab["e_ZP"] = e_ZP
    tab["e_ZP"].info.format = ".4f"

    plt.errorbar(
        pol(catalog["mag"] + ZP),
        catalog[band],
        xerr=catalog["e_mag"],
        yerr=catalog["e_" + band],
        marker="o",
        ls="",
        label=f"{band} (ZP = {ZP:.2f}+-{e_ZP:.2f})",
    )
    xx = [catalog[band].min(), catalog[band].max()]
    plt.plot(xx, xx, color="red", lw=1.0, label=f"1:1 ")
    plt.xlabel(f"Inst. photometry {band} [mag]")
    plt.ylabel(f"{vizier_catalog} {band} [mag]")
    plt.legend()
    plt.savefig(input_table.replace(".csv", "_plot.png"))
    plt.close()

    my_obj[band] = pol(my_obj["mag"] + ZP)
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
