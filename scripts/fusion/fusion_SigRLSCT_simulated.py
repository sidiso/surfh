import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import convolve2d as conv2
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import Angle
import udft

from surfh.Simulation import simulation_data
from surfh.DottestModels import SigRLSCT_Model
from surfh.ToolsDir import utils

from surfh.Simulation import fusion_CT
from surfh.Models import instru
from surfh.ToolsDir import fusion_mixing

chan_wavelength_axis = np.array([7.51065023, 7.51195023, 7.51325023, 7.51455023, 7.51585023,
       7.51715023, 7.51845023, 7.51975023, 7.52105023, 7.52235023,
       7.52365023, 7.52495023, 7.52625023, 7.52755023, 7.52885023,
       7.53015023, 7.53145023, 7.53275023, 7.53405023, 7.53535023,
       7.53665023, 7.53795023, 7.53925023, 7.54055023, 7.54185023,
       7.54315023, 7.54445023, 7.54575023, 7.54705023, 7.54835023,
       7.54965023, 7.55095023, 7.55225023, 7.55355023, 7.55485023,
       7.55615023, 7.55745023, 7.55875023, 7.56005023, 7.56135023,
       7.56265023, 7.56395023, 7.56525023, 7.56655023, 7.56785023,
       7.56915023, 7.57045023, 7.57175023, 7.57305023, 7.57435023,
       7.57565023, 7.57695023, 7.57825023, 7.57955023, 7.58085023,
       7.58215023, 7.58345023, 7.58475023, 7.58605023, 7.58735023,
       7.58865023, 7.58995023, 7.59125023, 7.59255023, 7.59385023,
       7.59515023, 7.59645023, 7.59775023, 7.59905023, 7.60035023,
       7.60165023, 7.60295023, 7.60425023, 7.60555023, 7.60685023,
       7.60815023, 7.60945023, 7.61075023, 7.61205023, 7.61335023,
       7.61465023, 7.61595023, 7.61725023, 7.61855023, 7.61985023,
       7.62115023, 7.62245023, 7.62375023, 7.62505023, 7.62635023,
       7.62765023, 7.62895023, 7.63025023, 7.63155023, 7.63285023,
       7.63415023, 7.63545023, 7.63675023, 7.63805023, 7.63935023,
       7.64065023, 7.64195023, 7.64325023, 7.64455023, 7.64585023,
       7.64715023, 7.64845023, 7.64975023, 7.65105023, 7.65235023,
       7.65365023, 7.65495023, 7.65625023, 7.65755023, 7.65885023,
       7.66015023, 7.66145023, 7.66275023, 7.66405023, 7.66535023,
       7.66665023, 7.66795023, 7.66925023, 7.67055023, 7.67185023,
       7.67315023, 7.67445023, 7.67575023, 7.67705023, 7.67835023,
       7.67965023, 7.68095023, 7.68225023, 7.68355023, 7.68485023,
       7.68615023, 7.68745023, 7.68875023, 7.69005023, 7.69135023,
       7.69265023, 7.69395023, 7.69525023, 7.69655023, 7.69785023,
       7.69915023, 7.70045023, 7.70175023, 7.70305023, 7.70435023,
       7.70565023, 7.70695023, 7.70825023, 7.70955023, 7.71085023,
       7.71215023, 7.71345023, 7.71475023, 7.71605023, 7.71735023,
       7.71865023, 7.71995023, 7.72125023, 7.72255023, 7.72385023,
       7.72515023, 7.72645023, 7.72775023, 7.72905023, 7.73035023,
       7.73165023, 7.73295023, 7.73425023, 7.73555023, 7.73685023,
       7.73815023, 7.73945023, 7.74075023, 7.74205023, 7.74335023,
       7.74465023, 7.74595023, 7.74725023, 7.74855023, 7.74985023,
       7.75115023, 7.75245023, 7.75375023, 7.75505023, 7.75635023,
       7.75765023, 7.75895023, 7.76025023, 7.76155023, 7.76285023,
       7.76415023, 7.76545023, 7.76675023, 7.76805023, 7.76935023,
       7.77065023, 7.77195023, 7.77325023, 7.77455023, 7.77585023,
       7.77715023, 7.77845023, 7.77975023, 7.78105023, 7.78235023,
       7.78365023, 7.78495023, 7.78625023, 7.78755023, 7.78885023,
       7.79015023, 7.79145023, 7.79275023, 7.79405023, 7.79535023,
       7.79665023, 7.79795023, 7.79925023, 7.80055023, 7.80185023,
       7.80315023, 7.80445023, 7.80575023, 7.80705023, 7.80835023,
       7.80965023, 7.81095023, 7.81225023, 7.81355023, 7.81485023,
       7.81615023, 7.81745023, 7.81875023, 7.82005023, 7.82135023,
       7.82265023, 7.82395023, 7.82525023, 7.82655023, 7.82785023,
       7.82915023, 7.83045023, 7.83175023, 7.83305023, 7.83435023,
       7.83565023, 7.83695023, 7.83825023, 7.83955023, 7.84085023,
       7.84215023, 7.84345023, 7.84475023, 7.84605023, 7.84735023,
       7.84865023, 7.84995023, 7.85125023, 7.85255023, 7.85385023,
       7.85515023, 7.85645023, 7.85775023, 7.85905023, 7.86035023,
       7.86165023, 7.86295023, 7.86425023, 7.86555023, 7.86685023,
       7.86815023, 7.86945023, 7.87075023, 7.87205023, 7.87335023,
       7.87465023, 7.87595023, 7.87725023, 7.87855023, 7.87985023,
       7.88115023, 7.88245023, 7.88375023, 7.88505023, 7.88635023,
       7.88765023, 7.88895023, 7.89025023, 7.89155023, 7.89285023,
       7.89415023, 7.89545023, 7.89675023, 7.89805023, 7.89935023,
       7.90065023, 7.90195023, 7.90325023, 7.90455023, 7.90585023,
       7.90715023, 7.90845023, 7.90975023, 7.91105023, 7.91235023,
       7.91365023, 7.91495023, 7.91625023, 7.91755023, 7.91885023,
       7.92015023, 7.92145023, 7.92275023, 7.92405023, 7.92535023,
       7.92665023, 7.92795023, 7.92925023, 7.93055023, 7.93185023,
       7.93315023, 7.93445023, 7.93575023, 7.93705023, 7.93835023,
       7.93965023, 7.94095023, 7.94225023, 7.94355023, 7.94485023,
       7.94615023, 7.94745023, 7.94875023, 7.95005023, 7.95135023,
       7.95265023, 7.95395023, 7.95525023, 7.95655023, 7.95785023,
       7.95915023, 7.96045023, 7.96175023, 7.96305023, 7.96435023,
       7.96565023, 7.96695023, 7.96825023, 7.96955023, 7.97085023,
       7.97215023, 7.97345023, 7.97475023, 7.97605023, 7.97735023,
       7.97865023, 7.97995023, 7.98125023, 7.98255023, 7.98385023,
       7.98515023, 7.98645023, 7.98775023, 7.98905023, 7.99035023,
       7.99165023, 7.99295023, 7.99425023, 7.99555023, 7.99685023,
       7.99815023, 7.99945023, 8.00075023, 8.00205023, 8.00335023,
       8.00465023, 8.00595023, 8.00725023, 8.00855023, 8.00985023,
       8.01115023, 8.01245023, 8.01375023, 8.01505023, 8.01635023,
       8.01765023, 8.01895023, 8.02025023, 8.02155023, 8.02285023,
       8.02415023, 8.02545023, 8.02675023, 8.02805023, 8.02935023,
       8.03065023, 8.03195023, 8.03325023, 8.03455023, 8.03585023,
       8.03715023, 8.03845023, 8.03975023, 8.04105023, 8.04235023,
       8.04365023, 8.04495023, 8.04625023, 8.04755023, 8.04885023,
       8.05015023, 8.05145023, 8.05275023, 8.05405023, 8.05535023,
       8.05665023, 8.05795023, 8.05925023, 8.06055023, 8.06185023,
       8.06315023, 8.06445023, 8.06575023, 8.06705023, 8.06835023,
       8.06965023, 8.07095023, 8.07225023, 8.07355023, 8.07485023,
       8.07615023, 8.07745023, 8.07875023, 8.08005023, 8.08135023,
       8.08265023, 8.08395023, 8.08525023, 8.08655023, 8.08785023,
       8.08915023, 8.09045023, 8.09175023, 8.09305023, 8.09435023,
       8.09565023, 8.09695023, 8.09825023, 8.09955023, 8.10085023,
       8.10215023, 8.10345023, 8.10475023, 8.10605023, 8.10735023,
       8.10865023, 8.10995023, 8.11125023, 8.11255023, 8.11385023,
       8.11515023, 8.11645023, 8.11775023, 8.11905023, 8.12035023,
       8.12165023, 8.12295023, 8.12425023, 8.12555023, 8.12685023,
       8.12815023, 8.12945023, 8.13075023, 8.13205023, 8.13335023,
       8.13465023, 8.13595023, 8.13725023, 8.13855023, 8.13985023,
       8.14115023, 8.14245023, 8.14375023, 8.14505023, 8.14635023,
       8.14765023, 8.14895023, 8.15025023, 8.15155023, 8.15285023,
       8.15415023, 8.15545023, 8.15675023, 8.15805023, 8.15935023,
       8.16065023, 8.16195023, 8.16325023, 8.16455023, 8.16585023,
       8.16715023, 8.16845023, 8.16975023, 8.17105023, 8.17235023,
       8.17365023, 8.17495023, 8.17625023, 8.17755023, 8.17885023,
       8.18015023, 8.18145023, 8.18275023, 8.18405023, 8.18535023,
       8.18665023, 8.18795023, 8.18925023, 8.19055023, 8.19185023,
       8.19315023, 8.19445023, 8.19575023, 8.19705023, 8.19835023,
       8.19965023, 8.20095023, 8.20225023, 8.20355023, 8.20485023,
       8.20615023, 8.20745023, 8.20875023, 8.21005023, 8.21135023,
       8.21265023, 8.21395023, 8.21525023, 8.21655023, 8.21785023,
       8.21915023, 8.22045023, 8.22175023, 8.22305023, 8.22435023,
       8.22565023, 8.22695023, 8.22825023, 8.22955023, 8.23085023,
       8.23215023, 8.23345023, 8.23475023, 8.23605023, 8.23735023,
       8.23865023, 8.23995023, 8.24125023, 8.24255023, 8.24385023,
       8.24515023, 8.24645023, 8.24775023, 8.24905023, 8.25035023,
       8.25165023, 8.25295023, 8.25425023, 8.25555023, 8.25685023,
       8.25815023, 8.25945023, 8.26075023, 8.26205023, 8.26335023,
       8.26465023, 8.26595023, 8.26725023, 8.26855023, 8.26985023,
       8.27115023, 8.27245023, 8.27375023, 8.27505023, 8.27635023,
       8.27765023, 8.27895023, 8.28025023, 8.28155023, 8.28285023,
       8.28415023, 8.28545023, 8.28675023, 8.28805023, 8.28935023,
       8.29065023, 8.29195023, 8.29325023, 8.29455023, 8.29585023,
       8.29715023, 8.29845023, 8.29975023, 8.30105023, 8.30235023,
       8.30365023, 8.30495023, 8.30625023, 8.30755023, 8.30885023,
       8.31015023, 8.31145023, 8.31275023, 8.31405023, 8.31535023,
       8.31665023, 8.31795023, 8.31925023, 8.32055023, 8.32185023,
       8.32315023, 8.32445023, 8.32575023, 8.32705023, 8.32835023,
       8.32965023, 8.33095023, 8.33225023, 8.33355023, 8.33485023,
       8.33615023, 8.33745023, 8.33875023, 8.34005023, 8.34135023,
       8.34265023, 8.34395023, 8.34525023, 8.34655023, 8.34785023,
       8.34915023, 8.35045023, 8.35175023, 8.35305023, 8.35435023,
       8.35565023, 8.35695023, 8.35825023, 8.35955023, 8.36085023,
       8.36215023, 8.36345023, 8.36475023, 8.36605023, 8.36735023,
       8.36865023, 8.36995023, 8.37125023, 8.37255023, 8.37385023,
       8.37515023, 8.37645023, 8.37775023, 8.37905023, 8.38035023,
       8.38165023, 8.38295023, 8.38425023, 8.38555023, 8.38685023,
       8.38815023, 8.38945023, 8.39075023, 8.39205023, 8.39335023,
       8.39465023, 8.39595023, 8.39725023, 8.39855023, 8.39985023,
       8.40115023, 8.40245023, 8.40375023, 8.40505023, 8.40635023,
       8.40765023, 8.40895023, 8.41025023, 8.41155023, 8.41285023,
       8.41415023, 8.41545023, 8.41675023, 8.41805023, 8.41935023,
       8.42065023, 8.42195023, 8.42325023, 8.42455023, 8.42585023,
       8.42715023, 8.42845023, 8.42975023, 8.43105023, 8.43235023,
       8.43365023, 8.43495023, 8.43625023, 8.43755023, 8.43885023,
       8.44015023, 8.44145023, 8.44275023, 8.44405023, 8.44535023,
       8.44665023, 8.44795023, 8.44925023, 8.45055023, 8.45185023,
       8.45315023, 8.45445023, 8.45575023, 8.45705023, 8.45835023,
       8.45965023, 8.46095023, 8.46225023, 8.46355023, 8.46485023,
       8.46615023, 8.46745023, 8.46875023, 8.47005023, 8.47135023,
       8.47265023, 8.47395023, 8.47525023, 8.47655023, 8.47785023,
       8.47915023, 8.48045023, 8.48175023, 8.48305023, 8.48435023,
       8.48565023, 8.48695023, 8.48825023, 8.48955023, 8.49085023,
       8.49215023, 8.49345023, 8.49475023, 8.49605023, 8.49735023,
       8.49865023, 8.49995023, 8.50125023, 8.50255023, 8.50385023,
       8.50515023, 8.50645023, 8.50775023, 8.50905023, 8.51035023,
       8.51165023, 8.51295023, 8.51425023, 8.51555023, 8.51685023,
       8.51815023, 8.51945023, 8.52075023, 8.52205023, 8.52335023,
       8.52465023, 8.52595023, 8.52725023, 8.52855023, 8.52985023,
       8.53115023, 8.53245023, 8.53375023, 8.53505023, 8.53635023,
       8.53765023, 8.53895023, 8.54025023, 8.54155023, 8.54285023,
       8.54415023, 8.54545023, 8.54675023, 8.54805023, 8.54935023,
       8.55065023, 8.55195023, 8.55325023, 8.55455023, 8.55585023,
       8.55715023, 8.55845023, 8.55975023, 8.56105023, 8.56235023,
       8.56365023, 8.56495023, 8.56625023, 8.56755023, 8.56885023,
       8.57015023, 8.57145023, 8.57275023, 8.57405023, 8.57535023,
       8.57665023, 8.57795023, 8.57925023, 8.58055023, 8.58185023,
       8.58315023, 8.58445023, 8.58575023, 8.58705023, 8.58835023,
       8.58965023, 8.59095023, 8.59225023, 8.59355023, 8.59485023,
       8.59615023, 8.59745023, 8.59875023, 8.60005023, 8.60135023,
       8.60265023, 8.60395023, 8.60525023, 8.60655023, 8.60785023,
       8.60915023, 8.61045023, 8.61175023, 8.61305023, 8.61435023,
       8.61565023, 8.61695023, 8.61825023, 8.61955023, 8.62085023,
       8.62215023, 8.62345023, 8.62475023, 8.62605023, 8.62735023,
       8.62865023, 8.62995023, 8.63125023, 8.63255023, 8.63385023,
       8.63515023, 8.63645023, 8.63775023, 8.63905023, 8.64035023,
       8.64165023, 8.64295023, 8.64425023, 8.64555023, 8.64685023,
       8.64815023, 8.64945023, 8.65075023, 8.65205023, 8.65335023,
       8.65465023, 8.65595023, 8.65725023, 8.65855023, 8.65985023,
       8.66115023, 8.66245023, 8.66375023, 8.66505023, 8.66635023,
       8.66765023, 8.66895023, 8.67025023, 8.67155023, 8.67285023,
       8.67415023, 8.67545023, 8.67675023, 8.67805023, 8.67935023,
       8.68065023, 8.68195023, 8.68325023, 8.68455023, 8.68585023,
       8.68715023, 8.68845023, 8.68975023, 8.69105023, 8.69235023,
       8.69365023, 8.69495023, 8.69625023, 8.69755023, 8.69885023,
       8.70015023, 8.70145023, 8.70275023, 8.70405023, 8.70535023,
       8.70665023, 8.70795023, 8.70925023, 8.71055023, 8.71185023,
       8.71315023, 8.71445023, 8.71575023, 8.71705023, 8.71835023,
       8.71965023, 8.72095023, 8.72225023, 8.72355023, 8.72485023,
       8.72615023, 8.72745023, 8.72875023, 8.73005023, 8.73135023,
       8.73265023, 8.73395023, 8.73525023, 8.73655023, 8.73785023,
       8.73915023, 8.74045023, 8.74175023, 8.74305023, 8.74435023,
       8.74565023, 8.74695023, 8.74825023, 8.74955023, 8.75085023,
       8.75215023, 8.75345023, 8.75475023, 8.75605023, 8.75735023,
       8.75865023, 8.75995023, 8.76125023, 8.76255023, 8.76385023,
       8.76515023, 8.76645023, 8.76775023, 8.76905023, 8.77035023])


"""
Create Model and simulation
"""
origin_alpha_axis, origin_beta_axis, wavel_axis, sotf, maps, templates = simulation_data.get_simulation_data() # subsampling to reduce dim of maps

step = 0.025 # arcsec
step_Angle = Angle(step, u.arcsec)

origin_alpha_width = origin_alpha_axis[-1] - origin_alpha_axis[0]
origin_beta_width = origin_beta_axis[-1] - origin_beta_axis[0]

origin_alpha_width_arcsec = origin_alpha_width*3600
origin_beta_width_arcsec = origin_beta_width*3600

grating_resolution = np.mean([2990, 3110])
spec_blur = instru.SpectralBlur(grating_resolution)

# Def Channel spec.
rchan = instru.IFU(
    fov=instru.FOV(4.0/3600, 4.8/3600, origin=instru.Coord(0, 0), angle=8.2),
    det_pix_size=0.196,
    n_slit=17,
    w_blur=spec_blur,
    pce=None,
    wavel_axis=chan_wavelength_axis,
    name="2A",
)

spectroModel = SigRLSCT_Model.spectroSigRLSCT(sotf, templates, origin_alpha_axis, origin_beta_axis, wavel_axis, rchan, step_Angle.degree)

y = spectroModel.forward(maps)
real_cube = spectroModel.mapsToCube(maps)


"""
Reconstruction method
"""
hyperParameter = 1e6
method = "lcg"
niter = 1000
value_init = 0

quadCrit_fusion = fusion_CT.QuadCriterion_MRS(mu_spectro=1, 
                                                    y_spectro=np.copy(y), 
                                                    model_spectro=spectroModel, 
                                                    mu_reg=hyperParameter, 
                                                    printing=True, 
                                                    gradient="separated"
                                                    )

res_fusion = quadCrit_fusion.run_method(method, niter, perf_crit = 1, calc_crit=True, value_init=value_init)


y_cube = spectroModel.mapsToCube(res_fusion.x)

utils.plot_maps(res_fusion.x)

y_adj = spectroModel.adjoint(y)
y_adj_cube = spectroModel.mapsToCube(y_adj)
utils.plot_3_cube(real_cube, y_cube, y_cube)

plt.figure()
xtick = np.arange(len(quadCrit_fusion.L_crit_val))*5
plt.plot(xtick, quadCrit_fusion.L_crit_val)
plt.yscale("log")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.show()

