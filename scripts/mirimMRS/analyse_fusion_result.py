import numpy as np
import os
import udft
from astropy.io import fits
from astropy.table import Table
import pathlib
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import Angle
from importlib import resources
import click
import statistics
import operator as op
from scipy.interpolate import interp1d

from surfh.Models import wavelength_mrs, realmiri, instru, spectroModel, MiriModel
from surfh.Simulation.fusion_CT import QuadCriterion_MRS
from surfh.Algorithm.criterion_spectroImageur import QuadCriterion_spectroImageur
from surfh.Vizualisation import cube_vizualisation
from surfh.ToolsDir import matrix_op, utils


def load_simulation_metadata_MRS(paths, step_angle, Npix):
    """Load simulation data."""
    imshape = (Npix, Npix)
    origin_alpha_axis = np.arange(imshape[0]) * step_angle
    origin_beta_axis = np.arange(imshape[1]) * step_angle
    origin_alpha_axis -= np.mean(origin_alpha_axis)
    origin_beta_axis -= np.mean(origin_beta_axis)
    
    spsf = np.load(os.path.join(paths['psf_dir'], 'psfs_pixscale0.1_npix_150_chan_1ABC_2ABC_3ABC_4ABC_SS4.npy'))

    sotf = udft.ir2fr(spsf, imshape)

    return origin_alpha_axis, origin_beta_axis, sotf


def load_simulation_metadata_MIRIM(paths, step_angle, Npix):
    """ Load simulation data."""
    imshape = (Npix, Npix)
    origin_alpha_axis = np.arange(imshape[0]) * step_angle
    origin_beta_axis = np.arange(imshape[1]) * step_angle
    origin_alpha_axis -= np.mean(origin_alpha_axis)
    origin_beta_axis -= np.mean(origin_beta_axis)
    
    spsf = np.load(os.path.join(paths['psf_dir'], 'mirim_psfs_pixscale0.1_npix_150.npy'))
    # sotf = udft.ir2fr(spsf, imshape)
    
    pce = np.load(os.path.join(paths['pce_path'], 'pce.npy'))
    print(pce.shape)
    return origin_alpha_axis, origin_beta_axis, spsf, pce

def create_ifus():
    """Create instrument configurations for each channel."""
    instruments = {}

    channel_specs = {
        '1a': (21, 3320, 3710, 0.196, 3.2/3600, 3.7/3600, 8.4),
        '1b': (21, 3190, 3750, 0.196, 3.2/3600, 3.7/3600, 8.4),
        '1c': (21, 3100, 3610, 0.196, 3.2/3600, 3.7/3600, 8.4),
        '2a': (17, 2990, 3110, 0.196, 4.0/3600, 4.8/3600, 8.1),
        '2b': (17, 2750, 3170, 0.196, 4.0/3600, 4.8/3600, 8.1),
        '2c': (17, 2860, 3300, 0.196, 4.0/3600, 4.8/3600, 8.1),
        '3a': (16, 2530, 2880, 0.245, 5.2/3600, 6.2/3600, 7.7),
        '3b': (16, 1790, 2640, 0.245, 5.2/3600, 6.2/3600, 7.7),
        '3c': (16, 1980, 2790, 0.245, 5.2/3600, 6.2/3600, 7.7),
        '4a': (12, 1460, 1930, 0.273, 6.6/3600, 7.7/3600, 8.3),
        '4b': (12, 1680, 1760, 0.273, 6.6/3600, 7.7/3600, 8.3),
        '4c': (12, 1630, 1330, 0.273, 6.6/3600, 7.7/3600, 8.3)
    }

    for chan, (n_slit, r_min, r_max, det_pix_size, fov_x, fov_y, rot_angle) in channel_specs.items():
        spec_blur = instru.SpectralBlur(np.mean([r_min, r_max]))
        instruments[chan] = instru.IFU(
            fov=instru.FOV(fov_x, fov_y, origin=instru.Coord(0, 0), angle=rot_angle),
            det_pix_size=det_pix_size,
            n_slit=n_slit,
            w_blur=spec_blur,
            pce=None,
            wavel_axis=wavelength_mrs.get_mrs_wavelength(chan),
            name=chan.upper()
        )

    return instruments


def create_spectroModel(sotf, templates, origin_alpha_axis, origin_beta_axis, wavel_axis, instruments, step_angle, pointings):
    """Create the spectrograph model."""
    print(f"Create spectroModel with Templates {type(templates)}")
    return spectroModel.spectroSigRLSCT(
        sotf=sotf,
        templates=templates,
        alpha_axis=origin_alpha_axis,
        beta_axis=origin_beta_axis,
        wavelength_axis=wavel_axis,
        instrs=list(instruments.values()),
        step_degree=step_angle, 
        pointings=pointings)

def load_skyModel(paths):
    def orion():
        """Rerturn maps, templates, spatial step and wavelength"""
        maps = fits.open(os.path.join(paths['template_dir'], "abundances_orion.fits"))[0].data

        h2_map = maps[0]
        if_map = maps[1]
        df_map = maps[2]
        mc_map = maps[3]

        spectrums = fits.open(os.path.join(paths['template_dir'], "spectra_mir_orion.fits"))[1].data
        wavel_axis = spectrums.wavelength

        h2_spectrum = spectrums["spectrum_h2"][: len(wavel_axis)]
        if_spectrum = spectrums["spectrum_if"][: len(wavel_axis)]
        df_spectrum = spectrums["spectrum_df"][: len(wavel_axis)]
        mc_spectrum = spectrums["spectrum_mc"][: len(wavel_axis)]

        return (
            np.asarray((h2_map, if_map, df_map, mc_map)),
            np.asarray([h2_spectrum, if_spectrum, df_spectrum, mc_spectrum]),
            wavel_axis,
        )
    maps, tpl, wavel_axis = orion()
    return maps, tpl, wavel_axis



def get_dithering(step_Angle, ifus):
    main_pointing = instru.Coord(0, 0)

    pointings = []
    
    ra_ref  = +0.0000
    dec_ref = -0.0007
    pix_res = 0.2/3600
    dec =  [ra_ref , ra_ref - 4.5*pix_res, ra_ref                 , ra_ref- 4.5*pix_res]
    ra = [dec_ref, dec_ref             , dec_ref + 4.5*pix_res, dec_ref + 4.5*pix_res]
    for idx, chan in enumerate(ifus.keys()):
        pointing_chan = [main_pointing + instru.Coord(ra[i], dec[i]) for i in range(4)]
        pointings.append(instru.CoordList(pointing_chan).pix(step_Angle))
    
    
    return pointings    


def initialize_parameters(fusion_dir_path, step=0.1):
    """Initialize global parameters."""
    paths = {
        'psf_dir': os.path.join(fusion_dir_path, 'PSF/'),
        'template_dir': os.path.join(fusion_dir_path, 'Templates/'),
        'save_filter_corrected_dir': os.path.join(fusion_dir_path, 'Filtered_slices/'),
        'result_path': os.path.join(fusion_dir_path, 'Results/'),
        'mask_path': os.path.join(fusion_dir_path, 'Masks/'),
        'pce_path': os.path.join(fusion_dir_path, 'PCE/')
    }
    step_angle = Angle(step, u.arcsec).degree

    return paths, step_angle

def reconstruction_method(imageurModel, mirim_data, spectroModel, mrs_data, result_path, hyperParameter, niter, method, bool_templates):
    """
    Perform the reconstruction method and save results.

    Parameters:
        spectroModel: Spectro model object
        ndata: Data array
        templates: Templates array
        pointings: Pointings data
        result_path: Path to save results
        wavel_axis: Wavelength axis array
    """
    # Hyperparameters
    # hyperParameter = 5e3
    # method = "lcg"
    # niter = 50
    value_init = 0

    # Create result directory
    result_dir = f'{method}_MC_{len(spectroModel.instrs)}_MO_4_lmm_{True}_nit_{str(niter)}_mu_{str("{:.2e}".format(hyperParameter))}/'
    path = pathlib.Path(result_path + result_dir)
    path.mkdir(parents=True, exist_ok=True)

    quadCrit_fusion = QuadCriterion_spectroImageur(
        mu_imager=1.0,
        y_imager=mirim_data,
        model_imager=imageurModel,
        mu_spectro=1.0,
        y_spectro=mrs_data,
        model_spectro=spectroModel,
        mu_reg=hyperParameter,
        printing=True,
        gradient='separated'
    )

    res_fusion = quadCrit_fusion.run_lcg(maximum_iterations=niter, 
                                         perf_crit=None, 
                                         calc_crit=True, 
                                         value_init=value_init)

    print(f"Results save in {path}")
    # Save results
    if bool_templates is False:
        print("No templates, save only cube")
        np.save(path / 'res_cube.npy', res_fusion.x)
        np.save(path / 'criterion.npy', quadCrit_fusion.L_crit_val_lcg)
    else:
        print("Templates loaded, save templates and cube")
        np.save(path / 'res_x.npy', res_fusion.x)
        np.save(path / 'criterion.npy', quadCrit_fusion.L_crit_val_lcg)
        np.save(path / 'res_cube.npy', spectroModel.mapsToCube(np.array(res_fusion.x)))

    utils.plot_maps(res_fusion.x)
    plt.show()


def get_MRS_slice(spectroModel, mrs_data, channel_idx, channel_name, wavelength):

    wavelength_idx = np.argmin(np.abs(wavelength_mrs.get_mrs_wavelength(channel_name) - wavelength))
    weighted_slice, _ = spectroModel.plot_slice(mrs_data, channel_idx, wavelength_idx)

    return weighted_slice


def plot_simulation_results(axs, axs2, spectroModel, sim_cube, res_cube, origin_alpha_axis, origin_beta_axis, wavel_axis, Npix, noisy_mirim_data, noisy_mrs_data):
    wavelengths = [6.5, 14, 21]
    channel_indices = [1, 7, 10]
    channel_name = ['1b', '3b', '4b']

    mirim_indices = [1, 5, 7]
    mirim_filters = ['F770W', 'F1500W', 'F2100W']

    

    for i, (wavelength, channel_idx, mirim_idx) in enumerate(zip(wavelengths, channel_indices, mirim_indices)):
        
        name = spectroModel.channels[channel_idx].instr.name
        wavel_idx = np.argmin(np.abs(wavel_axis - wavelength))
        mrs_slice = get_MRS_slice(spectroModel, noisy_mrs_data, channel_idx, channel_name[i], wavelength)

        plot_image(axs[i, 0], sim_cube[wavel_idx], origin_alpha_axis, origin_beta_axis, name, spectroModel.channels[channel_idx], title=f'$\lambda = {wavelength}\mu m$')
        plot_image(axs[i, 1], noisy_mirim_data[mirim_idx], origin_alpha_axis, origin_beta_axis, name, spectroModel.channels[channel_idx], legend=mirim_filters[i])
        plot_image(axs[i, 2], mrs_slice, origin_alpha_axis, origin_beta_axis, name, spectroModel.channels[channel_idx], legend=channel_name[i])
        plot_image(axs[i, 3], res_cube[wavel_idx], origin_alpha_axis, origin_beta_axis, name, spectroModel.channels[channel_idx])
        plot_image(axs[i, 4], sim_cube[wavel_idx] - res_cube[wavel_idx], origin_alpha_axis, origin_beta_axis, name, spectroModel.channels[channel_idx])
        axs[i, 3].plot(origin_alpha_axis[53],origin_beta_axis[77], '.', color='red')
        

        # plot_central_column(axs2[i], np.rot90(np.fliplr(res_cube[wavel_idx]), -1), sim_cube[wavel_idx], Npix, wavelength, spectroModel.channels[channel_idx])
        plot_central_column_with_FoV(axs2[i], np.rot90(np.fliplr(res_cube[wavel_idx]), -1), sim_cube[wavel_idx], Npix, wavelength, spectroModel.channels[channel_idx], origin_alpha_axis, origin_beta_axis)

import operator as op

def plot_image(ax, data, origin_alpha_axis, origin_beta_axis, name, chan, legend=None, title=''):
    im = ax.imshow(data, extent=[origin_alpha_axis[0], origin_alpha_axis[-1], origin_beta_axis[0], origin_beta_axis[-1]], cmap='viridis')

    # Définir une couleur pour les tracés
    plot_color = 'red'  # Vous pouvez choisir la couleur que vous préférez

    # Variable pour vérifier si le label a déjà été ajouté
    label_added = False

    # for i in range(4):
    #     fov = chan.instr.fov + chan.pointings[i]
    #     if not label_added:
    #         ax.plot(
    #             list(map(op.attrgetter("alpha"), fov.vertices)) + [fov.vertices[0].alpha],
    #             list(map(op.attrgetter("beta"), fov.vertices)) + [fov.vertices[0].beta],
    #             "-x",
    #             color=plot_color,
    #             linewidth=1,
    #             label=f'channel {name}',
    #         )
    #         label_added = True  # Le label a été ajouté
    #     else:
    #         ax.plot(
    #             list(map(op.attrgetter("alpha"), fov.vertices)) + [fov.vertices[0].alpha],
    #             list(map(op.attrgetter("beta"), fov.vertices)) + [fov.vertices[0].beta],
    #             "-x",
    #             color=plot_color,
    #             linewidth=1,
    #         )

    # ax.set_title(title)
    if legend is not None:
        ax.text(0.05, 0.95, legend, transform=ax.transAxes, bbox={'facecolor': 'white', 'pad': 10}, va='top', ha='left', fontsize=12)
    else:
        ax.text(-0.1, 0.5, title, va='center', ha='center', rotation='vertical', transform=ax.transAxes, fontsize=18)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.figure.colorbar(im, ax=ax)




def plot_central_column(ax, res_cube_slice, sim_cube_slice, Npix, wavelength, chan):
    ax.plot(res_cube_slice[:, Npix//2], label=r'$\widehat{x}$')
    ax.plot(sim_cube_slice[:, Npix//2], label=r'$x$')
    ax.set_title(f'Central Column $\lambda = {wavelength}\mu m$')
    ax.legend()

def plot_central_column_bis(ax, res_cube_slice, sim_cube_slice, Npix, wavelength, method, mu):
    ax.plot(res_cube_slice[:, Npix//2], label=r'$\widehat{x}$')
    ax.plot(sim_cube_slice[:, Npix//2], label=r'$x$')
    ax.set_ylabel(f'{method} $\mu = {mu}$')
    ax.legend()

def plot_central_column_with_FoV(ax, res_cube_slice, sim_cube_slice, Npix, wavelength, chan, origin_alpha_axis, origin_beta_axis):
    intersections = []
    alpha_coupe = origin_alpha_axis[Npix // 2]  

    fov = chan.instr.fov + chan.pointings[0]

    for j in range(len(fov.vertices)):
        alpha1, beta1 = fov.vertices[j].alpha, fov.vertices[j].beta
        alpha2, beta2 = fov.vertices[(j + 1) % len(fov.vertices)].alpha, fov.vertices[(j + 1) % len(fov.vertices)].beta

        if (alpha1 - alpha_coupe) * (alpha2 - alpha_coupe) < 0:  
            t = (alpha_coupe - alpha1) / (alpha2 - alpha1)
            beta_inter = beta1 + t * (beta2 - beta1)
            intersections.append(beta_inter)

    beta_axis = origin_beta_axis  # y-coordinates in physical units

    ax.plot(beta_axis, res_cube_slice[:, Npix//2], label=r'$\widehat{x}$')  
    ax.plot(beta_axis, sim_cube_slice[:, Npix//2], label=r'$x$')  
    for beta in intersections:
        ax.axvline(x=beta, color='red', linestyle='--', alpha=0.7, label="Intersection FoV" if beta == intersections[0] else "")

    ax.set_xlabel(r'$\beta$ coordinate') 
    ax.legend()


def add_gaussian_noise(signal, snr_db):
    """
    Ajoute un bruit gaussien à un signal donné pour obtenir un SNR spécifié en dB.

    :param signal: numpy array, le signal original
    :param snr_db: float, le rapport signal/bruit en dB
    :return: numpy array, le signal bruité
    """
    # Calcul de la puissance du signal
    signal_power = np.mean(signal**2)
    
    # Calcul de la puissance du bruit pour obtenir le SNR désiré
    noise_power = signal_power / (10**(snr_db / 10))
    
    # Génération du bruit gaussien
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    
    # Ajout du bruit au signal
    noisy_signal = signal + noise

    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)

    # Calculate SNR in linear scale
    snr_linear = signal_power / noise_power

    # Calculate SNR in decibels (dB)
    snr_db = 10 * np.log10(snr_linear)

    print("SNR (linear scale):", snr_linear)
    print("SNR (dB):", snr_db)


    return noisy_signal

def parse_options():

    fusion_dir = '/home/nmonnier/Data/JWST/Simulation/Orion/'
    result_dir = fusion_dir + 'Results/' + 'lcg_MC_12_MO_4_lmm_True_nit_2001_mu_5.00e+09'

    # Parameters
    step = 0.1 #arsec
    Npix_MRS = 150
    Npix_MIRIM = 150
    wavelength_ss = 4

    # Reconstruction parameters
    paths, step_angle = initialize_parameters(fusion_dir, step)

    # Load simulation data
    maps, tpl, wavel_axis    = load_skyModel(paths)
    tpl = tpl[:, ::wavelength_ss]
    wavel_axis = wavel_axis[::wavelength_ss]
    
    # MIRIM FoV selection
    maps = maps[:,155:305,588:738]

    print(f"shape maps {maps.shape}, shape tpl {tpl.shape}, shape wavel_axis {wavel_axis.shape}")



    alpha_axis_mrs, beta_axis_mrs, soft_mrs = load_simulation_metadata_MRS(paths, step_angle, Npix_MRS)
    alpha_axis_mirim, beta_axis_mirim, spsf_mirim, pce_mirim = load_simulation_metadata_MIRIM(paths, step_angle, Npix_MIRIM)
    # alpha_axis_mirim = np.flip(alpha_axis_mirim)
    # alpha_axis_mrs = np.flip(alpha_axis_mrs)
    beta_axis_mirim = np.flip(beta_axis_mirim)
    ifus = create_ifus()
    mrs_pointing = get_dithering(step_angle, ifus)
    spectroModel = create_spectroModel(soft_mrs, tpl, alpha_axis_mrs, beta_axis_mrs, wavel_axis, ifus, step_angle, mrs_pointing)
    try:
        H_freq = np.load(os.path.join(paths['template_dir'], "H_freq.npy"))
    except:
        H_freq = None
    imageurModel = MiriModel.Mirim_Model_LMM(spsf_mirim, pce_mirim, wavel_axis, tpl, (Npix_MIRIM, Npix_MIRIM), step, H_freq)
    print(f'Imageur ishape {imageurModel.ishape}, oshape {imageurModel.oshape}')
    


    plt.figure()
    for k in range(4):
        plt.plot(wavel_axis, tpl[k], label=f"tpl {k}")
    plt.xlabel(r'Wavelength $\lambda$', fontsize=14)
    plt.legend()
    plt.tight_layout()
    # plt.savefig('templates.png')


    # Créer une figure et une grille de sous-graphiques 2x2
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Aplatir le tableau axes pour faciliter l'itération
    axes = axes.flatten()
    
    # Parcourir chaque image et l'afficher avec sa colorbar
    for i in range(4):
        # Afficher l'image en conservant l'aspect ratio
        cax = axes[i].imshow(maps[i], 
                             extent=[alpha_axis_mirim[0], alpha_axis_mirim[-1], beta_axis_mirim[0], beta_axis_mirim[-1]], 
                             cmap='viridis',
                             )

        # Ajouter une colorbar
        fig.colorbar(cax, ax=axes[i], orientation='vertical', fraction=0.046, pad=0.04)

        # Ajouter un titre pour chaque image avec une taille agrandie
        axes[i].set_title(f'Carte d\'abondance n° {i+1}, MIRIM FoV', fontsize=16)

        # Définir les labels des axes
        axes[i].set_xlabel('RA (degrés)')
        axes[i].set_ylabel('DEC (degrés)')

        # # Limiter le nombre de ticks
        # axes[i].xaxis.set_major_locator(MaxNLocator(5))
        # axes[i].yaxis.set_major_locator(MaxNLocator(5))

        # Forcer l'aspect ratio à 1:1 pour éviter la déformation
        axes[i].set_aspect(1)
        label_added = False
        for j in range(4):
            if label_added == False:

                fov = spectroModel.channels[-1].instr.fov + spectroModel.channels[-1].pointings[j]
                axes[i].plot(
                    list(map(op.attrgetter("alpha"), fov.vertices)) + [fov.vertices[0].alpha],
                    list(map(op.attrgetter("beta"), fov.vertices)) + [fov.vertices[0].beta],
                    "-x",
                    label='MRS Ch4 FoV',
                    color='red'
                )
                label_added = True
            else:    
                fov = spectroModel.channels[-1].instr.fov + spectroModel.channels[-1].pointings[j]
                axes[i].plot(
                    list(map(op.attrgetter("alpha"), fov.vertices)) + [fov.vertices[0].alpha],
                    list(map(op.attrgetter("beta"), fov.vertices)) + [fov.vertices[0].beta],
                    "-x",
                    color='red'
                )
            axes[i].legend()

    # Ajuster la disposition
    plt.tight_layout()
    
    # plt.savefig('maps_FoV.png')
    # Afficher les images
    plt.show()



    # Simulate data
    mrs_data = spectroModel.forward(maps)
    mirim_data = imageurModel.forward(maps)
    print(f"Mirim data shape is {mirim_data.shape}")

    # Add noise to the data
    noisy_mirim_data = np.zeros_like(mirim_data)
    for i in range(mirim_data.shape[0]):
        noisy_mirim_data[i] = add_gaussian_noise(mirim_data[i], 30)

    noisy_mrs_data = np.zeros_like(mrs_data)
    for ch_idx, chan in enumerate(spectroModel.channels):
        reshape_spectro_data = mrs_data[spectroModel._idx[ch_idx] : spectroModel._idx[ch_idx + 1]].reshape(chan.oshape)
        noisy_mrs_data[spectroModel._idx[ch_idx] : spectroModel._idx[ch_idx + 1]] = add_gaussian_noise(reshape_spectro_data, 30).ravel()


    sim_cube = spectroModel.mapsToCube(maps)

    res_cube = np.load(result_dir + '/res_cube.npy')
    res_maps = np.load(result_dir + '/res_x.npy')
    criterion= np.load(result_dir + '/criterion.npy')

    fig, axs = plt.subplots(3, 5)
    fig2, axs2 = plt.subplots(3, 1)
    plot_simulation_results(axs, axs2, spectroModel, sim_cube, res_cube, alpha_axis_mrs, beta_axis_mrs, wavel_axis, Npix_MRS, noisy_mirim_data, noisy_mrs_data)
    # fig.savefig('simulation_results.png')


    # Premier point (53,73) pour la coupe --> (53,77) avec les coords
    plt.figure(figsize=(16, 6))  # Ajuster la taille de la figure pour plus de lisibilité

    # Tracer les données avec des lignes plus fines et des couleurs distinctes
    plt.plot(sim_cube[:, 75,67], label='True', linewidth=2, color='blue')
    plt.plot(res_cube[:, 75,67], label='Reconstructed', linewidth=2, color='red')
    plt.yscale('log')


    # Ajouter des titres et des étiquettes aux axes
    plt.title('Comparaison des données sim et res')
    plt.xlabel(r'Wavelength $\lambda$', fontsize=14)
    plt.ylabel(r'Intensity', fontsize=14)

    # Ajouter une grille
    plt.grid(True)

    # Ajouter une légende avec une taille de police plus grande
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('Coupe_wavel_1.png')
    plt.figure()
    plt.imshow(res_cube[53], cmap='viridis')
    plt.plot(75,67, '.', color='red')
    plt.title("Check coupe wavel")


    fig3, axs3 = plt.subplots(3, 4, figsize=(16, 8))  # Ajuster la taille de la figure pour plus de lisibilité

    # Ajuster l'espacement entre les sous-graphiques
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    # Tracer les données avec des titres et des barres de couleur
    for i in range(4):
        im1 = axs3[0, i].imshow(maps[i], cmap='viridis')
        axs3[0, i].set_title(r'True Map $a$ n° {}'.format(i+1))
        axs3[0, i].axis('off')  # Désactiver les axes si ce n'est pas nécessaire
        fig3.colorbar(im1, ax=axs3[0, i], orientation='vertical', fraction=0.046, pad=0.04)

        im2 = axs3[1, i].imshow(res_maps[i], cmap='viridis')
        axs3[1, i].set_title(r'Reconstructed Map $\widehat{{a}}$ n° {}'.format(i+1))
        axs3[1, i].axis('off')  # Désactiver les axes si ce n'est pas nécessaire
        fig3.colorbar(im2, ax=axs3[1, i], orientation='vertical', fraction=0.046, pad=0.04)

        im3 = axs3[2, i].imshow(maps[i]-res_maps[i], cmap='viridis')
        axs3[2, i].set_title(r'Diff Map $(a - \widehat{{a}})$ n° {}'.format(i+1))
        axs3[2, i].axis('off')  # Désactiver les axes si ce n'est pas nécessaire
        fig3.colorbar(im3, ax=axs3[2, i], orientation='vertical', fraction=0.046, pad=0.04)


    # Ajouter un titre global à la figure
    fig3.suptitle('True and reconstructed maps differences', fontsize=18)
    plt.tight_layout()
    # plt.savefig('maps_diff.png')



    plt.figure()
    plt.plot(criterion)
    plt.yscale('log')
                
   

    # # Define figure and subplots
    # fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    # axes = axes.flatten()
    # # Loop through the four maps
    # for i, ax in enumerate(axes):
    #     im = ax.imshow(
    #         maps[i],
    #         extent=[alpha_axis_mirim[0], alpha_axis_mirim[-1], beta_axis_mirim[0], beta_axis_mirim[-1]]
    #     )
    #     ax.set_title(f'Maps[{i}]')
    #     fig.colorbar(im, ax=ax)
        
    #     # Plot the field of view only for the first subplot
    #     for chan in spectroModel.channels:
    #         for pointing_idx in range(4):
    #             fov = chan.instr.fov + chan.pointings[pointing_idx]
    #             ax.plot(
    #                 [v.alpha for v in fov.vertices] + [fov.vertices[0].alpha],
    #                 [v.beta for v in fov.vertices] + [fov.vertices[0].beta],
    #                 '-x'
    #             )
    # plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parse_options()