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


class LMMChannel:
    """A channel with FOV, slit, spectral blurring and pce"""

    def __init__(
        self,
        instr: Instr,
        alpha_axis: array,
        beta_axis: array,
        wavel_axis: array,
        templates: array,
        spsf: array,
        srf: int,
        pointings: CoordList,
        ishape: "InputShape",
    ):
        self.alpha_axis = alpha_axis
        self.beta_axis = beta_axis
        if self.alpha_step != self.beta_step:
            logger.warning(
                "α and β step for input axis must be equal here α={da} and β={db}",
            )
        self.wavel_axis = wavel_axis

        self.pointings = pointings.pix(self.step)
        self.instr = instr.pix(self.step)

        # ∈ [0, β_s]
        beta_in_slit = np.arange(0, self.npix_slit) * self.beta_step
        # [β_idx, λ', λ]
        wpsf = self.instr.spectral_psf(
            beta_in_slit - np.mean(beta_in_slit),  # ∈ [-β_s / 2, β_s / 2]
            wavel_axis,
            # suppose that physical pixel detector are square. Therefor, moving
            # on sky by Δ_β = Δ_α in β direction should move a point source by
            # one physical pixel that is Δ_λ logical pixel. This factor make the
            # conversion.
            arcsec2micron=self.instr.wavel_step / self.instr.det_pix_size,
        )

        self.srf = srf
        self.imshape = (ishape.alpha, ishape.beta)
        logger.info(
            f"Precompute diffrated PSF {self.instr.name} with SRF {srf}",
        )
        # Make a λ slice because it is not necessary to compute diffracted psf
        # with all the input axis. Outside the λ' it is quickly negligeable.
        # margin is mandatory.
        wslices = self.instr.wslice(self.wavel_axis, margin=1)
        otf_np = np.asarray(
            [
                udft.ir2fr(
                    diffracted_psf(
                        tpl[wslices], spsf[wslices, ...], wpsf[..., wslices]
                    ),
                    self.imshape,
                )
                # * udft.ir2fr(np.ones((srf, 1)), self.imshape)[
                #     np.newaxis, np.newaxis, ...
                # ]  # * OTF for SuperResolution in alpha
                for tpl in templates
            ]
        )

        self.otf = xr.DataArray(otf_np, dims=("tpl", "beta", "wl_out", "nu_a", "nu_b"))

        self.local_alpha_axis, self.local_beta_axis = self.instr.fov.local_coords(
            self.step,
            alpha_margin=5 * self.step,
            beta_margin=5 * self.step,
        )
        self.ishape = (
            len(templates),
            len(alpha_axis),
            len(beta_axis),
        )
        self.cshape = (
            len(self.instr.wavel_axis),
            len(alpha_axis),
            len(beta_axis),
        )
        self.local_shape = (
            len(self.instr.wavel_axis),
            len(self.local_alpha_axis),
            len(self.local_beta_axis),
        )
        self.slit_shape = (
            len(self.instr.wavel_axis),
            self.n_alpha,
            self.npix_slit,
        )

    @property
    def name(self) -> str:
        return self.instr.name

    @property
    def step(self) -> float:
        return self.alpha_step

    @property
    def alpha_step(self) -> float:
        return self.alpha_axis[1] - self.alpha_axis[0]

    @property
    def beta_step(self) -> float:
        return self.beta_axis[1] - self.beta_axis[0]

    @property
    def npix_slit(self) -> int:
        """The number of pixel inside a slit"""
        return int(round(self.instr.slit_beta_width / self.beta_step))

    @property
    def n_alpha(self) -> int:
        return self.instr.fov.local.n_alpha(self.step)

    def slit_local_fov(self, num_slit: int) -> LocalFOV:
        slit_fov = self.instr.slit_fov[num_slit]
        return slit_fov.local + self.instr.slit_shift[num_slit]

    def slit_slices(self, num_slit: int) -> Tuple[slice, slice]:
        return self.slit_local_fov(num_slit).to_slices(
            self.local_alpha_axis, self.local_beta_axis
        )

    def slit_weights(self, num_slit: int) -> array:
        return fov_weight(
            self.slit_local_fov(num_slit),
            self.slit_slices(num_slit),
            self.local_alpha_axis,
            self.local_beta_axis,
        )[np.newaxis, ...]

    def slicing(
        self,
        gridded: array,
        num_slit: int,
    ) -> array:
        """Return a weighted slice of gridded. num_slit start at 0."""
        slices = self.slit_slices(num_slit)
        weights = self.slit_weights(num_slit)
        return gridded[:, slices[0], slices[1]] * weights

    def gridding(self, inarray, pointing):
        # α and β inside the FOV shifted to pointing, in the global ref.
        alpha_coord, beta_coord = (self.instr.fov + pointing).local2global(
            self.local_alpha_axis, self.local_beta_axis
        )

        # Necessary for interpn to process 3D array. No interpolation is done
        # along that axis.
        wl_idx = np.arange(inarray.shape[0])

        out_shape = (len(wl_idx),) + alpha_coord.shape

        local_coords = np.vstack(
            [
                np.repeat(
                    np.repeat(wl_idx.reshape((-1, 1, 1)), out_shape[1], axis=1),
                    out_shape[2],
                    axis=2,
                ).ravel(),
                np.repeat(alpha_coord[np.newaxis], out_shape[0], axis=0).ravel(),
                np.repeat(beta_coord[np.newaxis], out_shape[0], axis=0).ravel(),
            ]
        ).T

        # This output can be processed in local ref.
        return scipy.interpolate.interpn(
            (wl_idx, self.alpha_axis, self.beta_axis), inarray, local_coords
        ).reshape(out_shape)

    def forward(self, inarray):
        """inarray is supposed in self coordinate"""
        # [pointing, slit, λ', α]
        out = np.zeros(
            (
                len(self.pointings),
                self.instr.n_slit,
                len(self.instr.wavel_axis),
                # self.n_alpha // self.srf,
                ceil(self.n_alpha / self.srf),
            )
        )

        # duplicate in β
        for beta_idx in range(self.npix_slit):
            # Σ_tpl, duplicate in λ
            blurred = sum(
                abd[np.newaxis] * otf[beta_idx] for abd, otf in zip(inarray, self.otf)
            )
            blurred = idft(blurred, self.imshape)
            # Duplicate for each pointing
            for p_idx, pointing in enumerate(self.pointings):
                gridded = self.gridding(blurred, pointing)
                # Duplicate for each slit
                for num_slit in range(self.instr.n_slit):
                    # [λ', α]. Extract the result for this β in slit because
                    # it is the blur for this β
                    sliced = self.slicing(gridded, num_slit)[:, :, beta_idx]
                    # Σ_α for SR and Σ_β in slit
                    out[p_idx, num_slit, :, :] += np.add.reduceat(
                        sliced, range(0, sliced.shape[1], self.srf), axis=1
                    )[:, : out.shape[3]]

        return out

    def adjoint(self, measures):
        out = np.zeros(self.ishape)
        gridded = np.zeros(self.local_shape)
        blurredT = np.zeros(self.cshape)
        sliced = np.zeros(self.slit_shape)
        for beta_idx in range(self.npix_slit):
            blurredT.fill(0)
            for p_idx in pointing in enumerate(self.pointings):
                sliced.fill(0)
                # acc. slit
                for num_slit in range(self.instr.n_slit):
                    # β zero filling and duplicate α for SR
                    sliced[:, :, beta_idx] = np.repeat(
                        measures[p_idx, num_slit, :, :], self.srf, axis=1
                    )[:, self.n_alpha]

                    slices = self.slit_slices(num_slit)
                    assert (
                        slices[1].stop - slices[1].start == self.npix_slit
                    ), "The number of pixel in slit must corresponds to the slices size"

                    # acc. for slit
                    gridded[:, slices[0], slices[1]] += sliced * self.slit_weights(
                        num_slit
                    )
                # acc. pointing
                blurredT += self.degridding(gridded, pointing)

            # blurredT contains all pointing and specific beta of all slit
            blurredTf = dft(blurredT)
            # Duplicate for tpl and Σ_λ and Σ_β
            for tpl_idx in range(self.ishape[0]):
                out[tpl_idx] += idft(
                    np.sum(blurredTf * np.conj(otf[tpl_idx, beta_idx]), axis=0),
                    self.imshape,
                )

        return out
