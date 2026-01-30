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
from astropy.stats import sigma_clipped_stats, SigmaClip
import astropy.units as u
from photutils.aperture import (
    CircularAperture,
    CircularAnnulus,
    ApertureStats,
)
from photutils.background import (
    Background2D,
    SExtractorBackground,
)
from photutils.detection import DAOStarFinder
from photutils.profiles import RadialProfile
from astroquery.xmatch import XMatch
from astroquery.gaia import Gaia

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
        box_size: tuple = (10, 10),
        filter_size: tuple = (3, 3),
    ):

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

    def get_field_sources(
        self,
        threshold: float = 5.0,
        fwhm: float = 5.5,
        **kwargs,
    ):
        _, _, std = sigma_clipped_stats(self.data - self.data_bkg, sigma=3.0)
        starfinder = DAOStarFinder(
            threshold=threshold * std,
            fwhm=fwhm,
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
            self.sources["mjd"] = 0.0

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
        bkg_stats = ApertureStats(
            self.data - self.data_bkg, aper_annulus_pix, sigma_clip=sigclip
        )

        total_bkg = bkg_stats.median * aper_stats.sum_aper_area.value
        aper_sum_bksub = aper_stats.sum - total_bkg

        gain = 1.0
        if "GAIN" in self.hdr:
            gain = float(self.hdr["GAIN"])
        if "EXPTIME" not in self.hdr:
            self.hdr["EXPTIME"] = 1.0
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
    radius: float = 3.0,
    use_pol_fit: bool = False,
    obs_limits: dict | None = None,
) -> tuple:
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

    Returns
    -------
    None.

    """

    tab = Table.read(input_table, format="ascii.csv")

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

    catalog = XMatch.query(
        cat1=copy_tab,
        cat2=f"vizier:{vizier_catalog}",
        max_distance=radius * u.arcsec,
        colRA1="_RAJ2000",
        colDec1="_DEJ2000",
    )
    catalog["angDist"].name = "sep"

    if constrains is not None:
        print("Applying constrains to Vizier catalog...")
        for key in constrains.keys():
            print(f" - {key} {constrains[key]}")
            mask = eval(f"catalog['{key}'] {constrains[key]}")
            catalog = catalog[mask]

    mag_difs = np.asarray(catalog[band] - catalog["mag"])
    _, median_ZP, std_ZP = sigma_clipped_stats(
        mag_difs, sigma=3.0, maxiters=10, stdfunc="mad_std"
    )
    mask = np.abs(mag_difs - median_ZP) < (3.0 * std_ZP)
    ZP = np.nanmedian(mag_difs[mask])
    e_ZP = median_abs_deviation(mag_difs[mask], nan_policy="omit")

    print(f"ZP {band}: {ZP:.4f} +- {e_ZP:.4f}")

    pol = lambda x: x
    if use_pol_fit:
        x_fit = catalog["mag"] + ZP
        y_fit = catalog[band]
        weights = 1 / catalog["e_" + band]
        coeffs = np.polyfit(x_fit, y_fit, w=weights, deg=1)
        pol = np.poly1d(coeffs)

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

    return ZP, e_ZP


def calibrate_gaia(
    input_table: str,
    band: str,
    constrains: dict | None = {"e_Gmag": "<0.005"},
    radius: float = 2.0,
    obs_limits: dict | None = None,
):

    tab = Table.read(input_table, format="ascii.csv")

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

    catalog = XMatch.query(
        cat1=copy_tab,
        cat2="vizier:I/355/gaiadr3",
        max_distance=radius * u.arcsec,
        colRA1="_RAJ2000",
        colDec1="_DEJ2000",
    )

    print("Total GaiaDR3 sources: ", len(catalog))

    catalog["angDist"].name = "sep"
    catalog = catalog[["Source", "Gmag", "e_Gmag", "sep", "mag", "e_mag"]]

    if constrains is not None:
        print("Applying constrains to Gaia catalog...")
        for key in constrains.keys():
            print(f" - {key} {constrains[key]}")
            mask = eval(f"catalog['{key}'] {constrains[key]}")
            catalog = catalog[mask]

    catalog_name = input_table.replace(".csv", "_ref.csv")
    catalog.write(catalog_name, format="ascii.csv", overwrite=True)

    query = f"""
    SELECT my.*, gsyn.source_id, gsyn.c_star, gsyn.{band}_mag, gsyn.{band}_flag
    FROM gaiadr3.synthetic_photometry_gspc gsyn
    RIGHT JOIN tap_upload.catalog my
    ON my.Source = gsyn.source_id
    WHERE gsyn.{band}_flag = 1
    """

    job = Gaia.launch_job_async(
        query,
        upload_resource=catalog,
        upload_table_name="catalog",
        verbose=True,
    )

    synth_phot = job.get_results()

    mag_difs = np.asarray(synth_phot[f"{band}_mag"] - synth_phot["mag"])

    _, median_ZP, std_ZP = sigma_clipped_stats(
        mag_difs, sigma=3.0, maxiters=10, stdfunc="mad_std"
    )
    mask = np.abs(mag_difs - median_ZP) < (3.0 * std_ZP)
    ZP = np.nanmedian(mag_difs[mask])
    e_ZP = median_abs_deviation(mag_difs[mask], nan_policy="omit")

    print(f"ZP {band}: {ZP:.4f} +- {e_ZP:.4f}")

    plt.errorbar(
        synth_phot["mag"] + ZP,
        synth_phot[f"{band}_mag"],
        xerr=synth_phot["e_mag"],
        marker="o",
        ls="",
        label=f"{band} (ZP = {ZP:.2f}+-{e_ZP:.2f})",
    )
    xx = [synth_phot[f"{band}_mag"].min(), synth_phot[f"{band}_mag"].max()]
    plt.plot(xx, xx, color="red", lw=1.0, label=f"1:1 ")
    plt.xlabel(f"Inst. photometry {band} [mag]")
    plt.ylabel(f"Gaia Synthetic {band} [mag]")
    plt.legend()
    plt.savefig(input_table.replace(".csv", "_plot.png"))
    plt.close()

    return ZP, e_ZP
