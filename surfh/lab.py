#!/usr/bin/env python3


# class SpectralFilter:
#     def __init__(self, measured_wavelength, measured_values, name=""):
#         """A wavelength filter

#         Parameters
#         ==========
#         measured_wavelength: array-like
#           The wavelength where values is acquired, in meters

#         measured_values: array-like
#           The measured transmittance of the filter"""
#         self.measured_wavelength = measured_wavelength
#         self.measured_values = measured_values
#         self.name = name

#     def transmittance(self, wavelengths, normalized=False):
#         """Return the interpoled transmittance at given wavlengths"""
#         spectrum = np.interp(
#             wavelengths,
#             self.measured_wavelength,
#             self.measured_values,
#             left=self.measured_values[0],
#             right=self.measured_values[-1],
#         )
#         if normalized:
#             return spectrum / np.sum(spectrum)
#         else:
#             return spectrum

#     def integrate(self, cube, wavelength):
#         return sum(
#             image * weight
#             for image, weight in zip(cube, self.transmittance(wavelength, True))
#         )

#     def integrate_spectrum(self, spectrum, wavelength):
#         return np.sum(spectrum * self.transmittance(wavelength, True))

#     def draw(self, axe):
#         axe.plot(1e6 * self.measured_wavelength, self.measured_values)
#         axe.set_title(self.name)
#         axe.set_xlabel("Wavelength [um]")
#         axe.set_ylabel("Transmittance")
