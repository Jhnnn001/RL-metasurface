"""Small 2-D Meep model for a periodic slit filter."""
import numpy as np
import meep as mp


def spectrum(period, slit_width, metal_thickness, *, structure="slit", resolution=20,
             fcen=0.4, df=0.1, nfreq=3):
    if not (0 < slit_width <= period):
        raise ValueError("slit_width must be in (0, period]")
    if period <= 0 or metal_thickness <= 0:
        raise ValueError("period and metal_thickness must be positive")
    if structure not in {"air", "glass", "slit"}:
        raise ValueError("unknown structure")
    freqs = np.linspace(fcen - df / 2, fcen + df / 2, nfreq)
    if structure == "air":
        return {"frequency": freqs, "wavelength": 1 / freqs,
                "T": np.ones(nfreq), "R": np.zeros(nfreq)}
    if structure == "glass":
        r = ((1.0 - 1.5) / (1.0 + 1.5)) ** 2
        return {"frequency": freqs, "wavelength": 1 / freqs,
                "T": np.full(nfreq, 1 - r), "R": np.full(nfreq, r)}
    sx, sy, dpml = period, 8.0, 1.0
    cell = mp.Vector3(sx, sy)
    glass = mp.Medium(index=1.5)
    metal = mp.Medium(epsilon=1.0, D_conductivity=2.0)
    geometry = []
    if structure == "glass":
        geometry = [mp.Block(mp.Vector3(mp.inf, 2.0), center=mp.Vector3(0, 2), material=glass)]
    elif structure == "slit":
        geometry = [
            mp.Block(mp.Vector3(mp.inf, 2.0), center=mp.Vector3(0, 2), material=glass),
            mp.Block(mp.Vector3(mp.inf, metal_thickness), center=mp.Vector3(0, 0), material=metal),
            mp.Block(mp.Vector3(period - slit_width, metal_thickness), center=mp.Vector3(0, 0), material=metal),
        ]
    pml = [mp.PML(dpml, direction=mp.Y)]
    src = [mp.Source(mp.GaussianSource(fcen, fwidth=df), component=mp.Ez,
                     center=mp.Vector3(0, -0.5 * sy + dpml), size=mp.Vector3(sx, 0))]
    sim = mp.Simulation(cell_size=cell, boundary_layers=pml, geometry=geometry,
                        sources=src, resolution=resolution)
    flux = sim.add_flux(fcen, df, nfreq,
                        mp.FluxRegion(center=mp.Vector3(0, 0.5 * sy - dpml),
                                      size=mp.Vector3(sx, 0), direction=mp.Y))
    sim.run(until=120)
    values = np.asarray(mp.get_fluxes(flux), float)
    freqs = np.asarray(mp.get_flux_freqs(flux), float)
    return {"frequency": freqs, "wavelength": 1 / freqs, "T": values,
            "R": np.full_like(values, np.nan)}
