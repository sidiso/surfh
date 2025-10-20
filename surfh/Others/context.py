from dataclasses import dataclass
import yaml
from typing import List
from pathlib import Path


@dataclass
class ConfigurationPaths:
    fusion_dir: str
    data_dir: str
    result_dir: str
    psf_dir: str
    mrs_psf_file : str
    template_dir: str
    templates_file: str
    wavelength_file: str
    reference_wavelength_file: str

    def resolve_paths(self) -> None:
        """Combine tous les chemins relatifs avec fusion_dir, en gérant PSF et Templates correctement."""
        base = Path(self.fusion_dir).expanduser().resolve()

        for name in ["data_dir", "result_dir", "psf_dir", "template_dir"]:
            value = getattr(self, name)
            p = Path(value)
            if not p.is_absolute():
                p = base / p
            setattr(self, name, str(p))

        psf_file = Path(self.mrs_psf_file)
        if not psf_file.is_absolute():
            psf_file = Path(self.psf_dir) / psf_file
        self.mrs_psf_file = str(psf_file)

        for name in ["templates_file", "wavelength_file", "reference_wavelength_file"]:
            value = getattr(self, name)
            f = Path(value)
            if not f.is_absolute():
                f = Path(self.template_dir) / f
            setattr(self, name, str(f))

        self.fusion_dir = str(base)

@dataclass
class CubeConfig:
    npix: int
    pixel_resolution: float


@dataclass
class ReconstructionConfig:
    method: str
    max_iter: int
    mu: float

    def resolve_type(self) -> None:
        self.mu = float(self.mu)

@dataclass
class MRSConfig:
    list_channels: List[str]
    scale_data: bool
    spectral_lines: bool


@dataclass
class Config:
    configuration: ConfigurationPaths
    cube: CubeConfig
    reconstruction: ReconstructionConfig
    MRS: MRSConfig

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Lit un fichier YAML et renvoie une instance de Config."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        cfg = cls(
            configuration=ConfigurationPaths(**data["configuration"]),
            cube=CubeConfig(**data["cube"]),
            reconstruction=ReconstructionConfig(**data["reconstruction"]),
            MRS=MRSConfig(**data["MRS"]),
        )

        # Résolution des chemins relatifs
        cfg.configuration.resolve_paths()
        cfg.reconstruction.resolve_type()

        return cfg

    def validate_paths(self) -> None:
        """Valide l’existence des répertoires et fichiers configurés."""
        for name, value in vars(self.configuration).items():
            p = Path(value)
            print(f"Vérification de {name} : {p}")
            if name.endswith("_dir"):
                if not p.is_dir():
                    print(f"⚠️  Dossier manquant : {p}")
            elif name.endswith("_file"):
                if not p.is_file():
                    print(f"⚠️  Fichier manquant : {p}")