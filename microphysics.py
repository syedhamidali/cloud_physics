import numpy as np
import xarray as xr

# Physical constants
R = 287.053  # Specific gas constant for dry air in J/(kg*K)
Rv = 461.5   # Specific gas constant for water vapor in J/(kg*K)
g = 9.81     # Gravitational acceleration in m/s^2
cp = 1005    # Specific heat at constant pressure for dry air in J/(kg*K)
Lv = 2.5e6   # Latent heat of vaporization in J/kg

def get_air_density(pressure, temperature):
    """
    Calculate air density using the ideal gas law.

    Parameters
    ----------
    pressure : np.array or xr.DataArray
        Air pressure field in Pa.
    temperature : np.array or xr.DataArray
        Temperature field in K.

    Returns
    -------
    np.array or xr.DataArray
        Air density for a dry air mass in kg/m^3.
    """
    rho_dry = pressure / (R * temperature)
    return rho_dry

def mixing_ratio_to_density(mixing_ratio, air_density):
    """
    Convert the mixing ratio of a substance to its density.

    Parameters
    ----------
    mixing_ratio : np.array or xr.DataArray
        Field with mixing ratio of a hydrometeor in kg/kg.
    air_density : np.array or xr.DataArray
        Field with air density in kg/m^3.

    Returns
    -------
    np.array or xr.DataArray
        Density field of the hydrometeor in kg/m^3.
    """
    density = mixing_ratio * air_density
    return density

def vapor_pressure_to_mixing_ratio(vapor_pressure, pressure):
    """
    Convert vapor pressure to mixing ratio.

    Parameters
    ----------
    vapor_pressure : np.array or xr.DataArray
        Vapor pressure of water in Pa.
    pressure : np.array or xr.DataArray
        Total air pressure in Pa.

    Returns
    -------
    np.array or xr.DataArray
        Mixing ratio in kg/kg.
    """
    mixing_ratio = (R / Rv) * (vapor_pressure / (pressure - vapor_pressure))
    return mixing_ratio

def get_dqs_des(vapor_pressure, pressure):
    """
    Calculate the derivative of saturation mixing ratio with respect to saturation vapor pressure.

    Parameters
    ----------
    vapor_pressure : np.array or xr.DataArray
        Vapor pressure of water in Pa.
    pressure : np.array or xr.DataArray
        Total air pressure in Pa.

    Returns
    -------
    np.array or xr.DataArray
        Derivative of saturation mixing ratio with respect to vapor pressure.
    """
    dqs_des = (R / Rv) * (pressure / (pressure - vapor_pressure) ** 2)
    return dqs_des

def pressure_integration(mixing_ratio, pressure, axis=0):
    """
    Integrate mixing ratio over pressure.

    Parameters
    ----------
    mixing_ratio : np.array or xr.DataArray
        Field of mixing ratio, where one dimension is pressure levels.
    pressure : np.array or xr.DataArray
        Pressure levels in Pa.
    axis : int, optional
        Axis along which to integrate, by default 0.

    Returns
    -------
    np.array or xr.DataArray
        Integrated mixing ratio in kg/m^2.
    """
    integrated_mass = np.trapz(mixing_ratio, pressure, axis=axis) * (1 / g)
    return integrated_mass

def height_integration(density, heights, axis=0):
    """
    Integrate a quantity (e.g., density) over height.

    Parameters
    ----------
    density : np.array or xr.DataArray
        Field of hydrometeor quantity as density or concentration (kg/m^3).
    heights : np.array or xr.DataArray
        Height levels in meters.
    axis : int, optional
        Axis along which to integrate, by default 0.

    Returns
    -------
    np.array or xr.DataArray
        Integrated quantity in kg/m^2.
    """
    integrated_mass = np.trapz(density, heights, axis=axis)
    return integrated_mass

def get_saturation_vapor_pressure(temperature):
    """
    Estimate the saturation vapor pressure using the August-Roche-Magnus approximation.

    Parameters
    ----------
    temperature : np.array or xr.DataArray
        Temperature field in K.

    Returns
    -------
    np.array or xr.DataArray
        Saturation vapor pressure in Pa.
    """
    temp_celsius = temperature - 273.15
    es = 6.1094 * np.exp(17.625 * temp_celsius / (temp_celsius + 243.04))
    es_Pa = es * 100
    return es_Pa

def get_des_dT(temperature, es):
    """
    Calculate the rate of change of saturation vapor pressure with temperature.

    Parameters
    ----------
    temperature : np.array or xr.DataArray
        Temperature field in K.
    es : np.array or xr.DataArray
        Saturation vapor pressure in Pa.

    Returns
    -------
    np.array or xr.DataArray
        Change rate of saturation vapor pressure with temperature in Pa/K.
    """
    des_dT = Lv * es / (Rv * temperature ** 2)
    return des_dT

def get_condensation_rate(vertical_velocity, temperature, pressure):
    """
    Estimate the condensation rate based on saturation adjustment.

    Parameters
    ----------
    vertical_velocity : np.array or xr.DataArray
        Vertical velocity field in m/s.
    temperature : np.array or xr.DataArray
        Temperature field in K.
    pressure : np.array or xr.DataArray
        Air pressure field in Pa.

    Returns
    -------
    np.array or xr.DataArray
        Condensation rate in kg/kg/s.
    """
    es = get_saturation_vapor_pressure(temperature)
    qs = vapor_pressure_to_mixing_ratio(es, pressure)
    des_dT = get_des_dT(temperature, es)
    rho = get_air_density(pressure, temperature)
    dqs_des = get_dqs_des(es, pressure)
    dqs_dT = des_dT * dqs_des

    condensation_rate = (
        g * vertical_velocity * 
        (dqs_dT * cp ** (-1) - (qs * rho) / (pressure - es)) *
        (1 + dqs_dT * (Lv / cp)) ** (-1)
    )

    return condensation_rate
