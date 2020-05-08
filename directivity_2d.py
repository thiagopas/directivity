import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import makevideo
import geometry
from scipy import signal

###### User configurable parameters ############################
######                              ############################

# Transducer's width [mm]
width = .5
# Sampling frequency [Hz]
fs = 5e10
# Speed of sound [m/s]
c_m_s = 1485.0
# Minimum size (samples) of the zero-padded SIR, both in time and frequency domains
sir_size_samples_min = 1e6 + 1
# Exhibit animation along time
anim_time = False
# Exhibit animation along angles
anim_angle = True
# Monochromatic or excitation pulse
monochromatic = True
# Scatterer angles (degrees)
scat_angles = np.linspace(0, 90, 91)
# Distance from scatterer to transducer center
scat_distances = np.linspace(25, 10000, 1)
# Frequency (monochromatic excitation) [MHz]
freq_monochr = 5.
# Gain (at freq_monochr) per angle
gains_monochr_angle = np.zeros_like(scat_angles)
# Excitation pulse
t_excit = np.arange(0, sir_size_samples_min / fs, 1 / fs)
t_excit = t_excit - t_excit[int(t_excit.size / 2)]
excit_pulse = signal.gausspulse(t_excit, fc=freq_monochr * 1e6, bw=0.5)
excit_fft = np.fft.fft(excit_pulse)
excit_fft = np.fft.fftshift(excit_fft)
indices6dB = np.where(np.abs(excit_fft) >= np.max(np.abs(excit_fft)) / 2)
indices6dB_left = indices6dB[0][0:int(indices6dB[0].size / 2)]
indices6dB_right = indices6dB[0][int(indices6dB[0].size / 2):]
indices_lvls = list()
alphas_lvls = list()
max_excit_fft = np.max(np.abs(excit_fft))
num_levels = 25
if monochromatic is False:
    for i_levels in range(1, num_levels):
        indices_lvl = np.where((max_excit_fft * (i_levels - 1) / num_levels) <=
                               (np.abs(excit_fft)))
        indices_lvls.append(indices_lvl[0][0:int(indices_lvl[0].size / 2)])
        indices_lvls.append(indices_lvl[0][int(indices_lvl[0].size / 2):])
        alphas_lvls.append(i_levels / num_levels)
        alphas_lvls.append(i_levels / num_levels)

################################################################
################################################################
stored_SIRs = list()


def correlation(u, v):
    if u.size > v.size:
        v = np.append(v, np.zeros(u.size - v.size))
    elif v.size > u.size:
        u = np.append(u, np.zeros(v.size - u.size))
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


# x coordinate of tranducer's left corner [mm]
xcd_left = 0.0
# Speed of sound [mm/s]
c = c_m_s * 1000.0
# x coordinate of transducer's right corner [mm]
xcd_right = xcd_left + width
# x coordinate of transducer's center [mm]
xcd_center = xcd_left + width / 2
# Size (samples) of the zero-padded SIR, both in time and frequency domains
sir_size_samples = np.int(np.max([sir_size_samples_min, 1.1 * (xcd_right - xcd_right) / (c * fs)]))
if sir_size_samples % 2 != 1:
    sir_size_samples += 1
# Wavelength [mm]
wavelen = c / (freq_monochr * 1e6)
# Near field distance (Fraunhofer distance) [mm]
fraunhofer_d = np.pi * (width / 2) ** 2 / wavelen
sir_momentum = width / c

np.seterr(divide='ignore')
makevideo.delfigs()

subplot_gainperangle = None
SIR_per_distance = list()
for i_dist in range(scat_distances.size):
    scat_distance = scat_distances[i_dist]
    # Approximate time of flight from scatterer to transducer
    approx_tof = scat_distance / c
    SIR_per_angle = list()
    for i_angle in range(scat_angles.size):
        if anim_angle is True:
            plt.clf()
        current_angle = scat_angles[i_angle]
        # Scatterer's position (x, z) [mm]
        scat = np.array([scat_distance * np.sin(current_angle * np.pi / 180) + xcd_center,
                         scat_distance * np.cos(current_angle * np.pi / 180)])
        important_times = list()
        important_times.append(np.sqrt((scat[0] - xcd_left) ** 2 + scat[1] ** 2))
        important_times.append(np.sqrt((scat[0] - xcd_right) ** 2 + scat[1] ** 2))
        if xcd_left <= scat[0] <= xcd_right:
            important_times.append(scat[1])
        important_times = np.array(important_times)
        important_times = important_times / c
        # Simulation initial time [s]
        t_init = 0.0
        # Simulation final time
        t_final = important_times.max() + 2 / fs
        # Simulation toggle time resolution
        t_HR_threshold = important_times.min()

        # Time array
        t_lr = np.arange(t_init, t_HR_threshold, 1e-6)
        t_hr = np.arange(t_HR_threshold, t_final, 1 / fs)
        if t_hr.size < 3:
            t_hr = np.arange(t_HR_threshold, t_HR_threshold + 3 / fs, 1 / fs)
        t = t_hr
        t = np.append(t, 2 * t[-1] - t[-2])
        # Spatial impulse response
        sir = np.zeros_like(t)

        # Angles array (to draw circle)
        angles = np.arange(0, 2 * np.pi, .01)
        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)

        lim_obj = geometry.LimitsObjects()
        lim_obj.set_x_min(np.min([xcd_left, scat[0]]))
        lim_obj.set_x_max(np.max([xcd_right, scat[0]]))
        lim_obj.set_z_min(0)
        lim_obj.set_z_max(scat[1])
        lim_obj.calc_axis()




        if anim_time is True:
            i = 0
            for t_now in t:
                arc_total = 0
                # Compute current radius
                r = c * t_now

                # Compute current radius of the projection of the wavefront
                # on the plane z=0.
                # Geometric scheme: https://photos.app.goo.gl/N6HJ6SSz3kSYaYiH9
                if r >= scat[1]:
                    z_proj_exists = True
                    beta = np.arccos(scat[1] / r)
                    r_top = r * np.sin(beta)
                else:
                    z_proj_exists = False

                # Compute instersection between transducer and wavefront (front view)
                a = 1
                b = -2 * scat[0]
                C = scat[0] ** 2 + scat[1] ** 2 - r ** 2
                delta = b ** 2 - 4 * a * C
                if delta < 0:
                    intersection_1 = False
                    intersection_2 = False
                else:
                    x_1 = (-1 * b + np.sqrt(delta)) / (2 * a)
                    x_2 = (-1 * b - np.sqrt(delta)) / (2 * a)
                    if (x_1 >= xcd_left) and (x_1 <= (xcd_left + width)):
                        intersection_1 = True
                    else:
                        intersection_1 = False
                    if (x_2 >= xcd_left) and (x_2 <= (xcd_left + width)):
                        intersection_2 = True
                    else:
                        intersection_2 = False

                # Clear figure
                if anim_time is True:
                    plt.clf()

                ####################### Front view
                if i == t.size - 1 or anim_time is True:
                    plt.subplot(2, 2, 1)
                    # Draw line z=0
                    plt.plot([-1e10, 1e10], [0, 0], c='gray', linewidth=.5)
                    plt.plot([xcd_left, xcd_left], [1e10, 1e-10], c='gray', linewidth=.5)
                    plt.plot([xcd_right, xcd_right], [1e10, 1e-10], c='gray', linewidth=.5)
                    # Draw transducer's surface
                    plt.plot([xcd_left, width], [0, 0], linewidth=2)
                    # Draw scatterer
                    plt.scatter([scat[0]], [scat[1]], c='k')
                if anim_time is True:
                    # Draw circle
                    circle_x = scat[0] + r * cos_angles
                    circle_z = scat[1] + r * sin_angles
                    plt.plot(circle_x, circle_z, 'k')
                if i == t.size - 1 or anim_time is True:
                    plt.title('Front view (y=0) ')  # + str(scat[0]))
                    plt.xlabel('x [mm]')
                    plt.ylabel('z [mm]')
                    plt.axis('square')
                    plt.axis(lim_obj.get_axis_xz())
                    # Mirror vertical axis
                    ax = plt.gca()
                    ax.set_ylim(ax.get_ylim()[::-1])
                    plt.draw()
                # Draw intersection point (if exists)
                # if intersection_1 == True:
                #    plt.scatter([x_1], [0], c='r')
                # if intersection_2 == True:
                #    plt.scatter([x_2], [0], c='r')

                ####################### Top view
                if subplot_gainperangle is None:
                    subplot_gainperangle = plt.subplot(2, 2, 3)
                if anim_time is True:
                    # Draw transducer's surface
                    xdc_rect = patches.Rectangle([xcd_left, -1e3], width=width, height=2e3, fill=True,
                                                 color='blue')
                    # Draw scatterer
                    plt.subplot(subplot_gainperangle)
                    plt.scatter([scat[0]], [0], c='k')
                    ax = plt.gca()
                    ax.add_patch(xdc_rect)

                # Draw lines (lateral limits of the cilinder)
                if z_proj_exists:
                    # Left line
                    left_line_x = scat[0] - r_top
                    if xcd_left <= left_line_x <= xcd_right:
                        # sir[i] += np.divide(1, np.sin(np.arctan(np.divide(r_top, scat[1]))))/sir_momentum
                        # sir[i] += np.divide(1, np.sin(np.arccos(np.divide(scat[1], c*t_now)))) / sir_momentum
                        sir[i] += sir_integral[i + 1] - sir_integral[i]
                        # if sir[i] > fs:
                        # sir[i] = fs
                        if anim_time is True:
                            plt.plot([left_line_x, left_line_x], [-1e6, 1e6], c='r')
                    else:
                        if anim_time is True:
                            plt.plot([left_line_x, left_line_x], [-1e6, 1e6], c='k')
                    # Right line
                    right_line_x = scat[0] + r_top
                    if xcd_left <= right_line_x <= xcd_right:
                        # sir[i] += np.divide(1, np.sin(np.arctan(np.divide(r_top, scat[1]))))/sir_momentum
                        # sir[i] += np.divide(1, np.sin(np.arccos(np.divide(scat[1], c * t_now)))) / sir_momentum
                        sir[i] += sir_integral[i + 1] - sir_integral[i]
                        # if sir[i] > fs:
                        # sir[i] = fs
                        if anim_time is True:
                            plt.plot([right_line_x, right_line_x], [-1e6, 1e6], c='r')
                    else:
                        if anim_time is True:
                            plt.plot([right_line_x, right_line_x], [-1e6, 1e6], c='k')

                if anim_time is True:
                    plt.title('Top view (z=0)')
                    plt.xlabel('x[mm]')
                    plt.ylabel('y[mm]')
                    plt.axis('square')
                    plt.axis(lim_obj.get_axis_xy())

                ####################### Spatial impulse response
                if i == t.size - 1 or anim_time is True:
                    plt.subplot(2, 2, 2)
                    plt.plot(t[:i] * 1e6, sir[:i], '-o', linewidth=2)
                    # plt.axis([(approx_tof-.75*width/c)*1e6, (approx_tof+.75*width/c)*1e6, -.01*sir.max(), 1.1*sir.max()])
                    # plt.axis([(approx_tof - .75 * width / c) * 1e6, (approx_tof + .75 * width / c) * 1e6, 0, .25e7])
                    # plt.axis([6, 8, -.2, 2])
                    # plt.axis([approx_tof * 1e6 - 1, approx_tof * 1e6 + 1, -.2, 2])
                    # plt.axis([t[0] * 1e6, t[-1] * 1e6, -.2, 1.2])
                    plt.xlabel('time [μs]')
                    plt.grid()
                    plt.title('SIR')
                    ax = plt.axis()
                    plt.axis([(scat_distances[i_dist]-xcd_right-width/4)*1e6/c, (scat_distances[i_dist]-xcd_left+width)*1e6/c, 0, ax[3]*1.1])
                    mng = plt.get_current_fig_manager()
                    mng.resize(700, 870)
                if anim_time is True:
                    plt.savefig('./figs/' + "%05d" % (i,) + '.png')

                i += 1
        else:
            ####################### Front view
            plt.subplot(2, 2, 1)
            # Draw line z=0
            plt.plot([-1e10, 1e10], [0, 0], c='gray', linewidth=.5)
            plt.plot([xcd_left, xcd_left], [1e10, 1e-10], c='gray', linewidth=.5)
            plt.plot([xcd_right, xcd_right], [1e10, 1e-10], c='gray', linewidth=.5)
            # Draw transducer's surface
            plt.plot([xcd_left, width], [0, 0], linewidth=2)
            # Draw scatterer
            plt.scatter([scat[0]], [scat[1]], c='k')
            plt.title('Front view (y=0) ')  # + str(scat[0]))
            plt.xlabel('x [mm]')
            plt.ylabel('z [mm]')
            plt.axis('square')
            plt.axis(lim_obj.get_axis_xz())
            # Mirror vertical axis
            ax = plt.gca()
            ax.set_ylim(ax.get_ylim()[::-1])
            plt.axis([-2, scat_distances[i_dist]+2, scat_distances[i_dist]+2, -2])
            plt.draw()

            ####################### Top view
            if subplot_gainperangle is None:
                subplot_gainperangle = plt.subplot(2, 2, 3)

            ####################### Compute Spatial impulse response
            # Adjust time offset
            if scat[0] <= xcd_right:    # If scatterer is under the element
                t_0 = scat[1] / c
            else:   # If scatterer is not under element (it is at right)
                t_0 = np.sqrt((scat[0]-xcd_right)**2 + scat[1]**2) / c
            t = t - (t[0] - t_0)

            # Compute integral-based SIR (more stable)
            sqrt_arg = 1 - np.divide(scat[1] ** 2, (c ** 2) * (t ** 2))
            sqrt_arg = sqrt_arg * (sqrt_arg >= 0)
            sir_integral = t * np.sqrt(sqrt_arg)
            sir_integ_diff = np.zeros_like(sir_integral)
            sir_integ_diff[0:-1] = np.diff(sir_integral) / fs
            sir_integ_diff[-1] = sir_integ_diff[-2]

            print(i_angle)
            r_top_squared = c ** 2 * t ** 2 - scat[1] ** 2
            r_top_squared[np.where((r_top_squared < 0) * (-1e-5 < r_top_squared))] = 0
            r_geq_z = r_top_squared >= 0
            r_top_vec = np.zeros_like(t)
            r_top_vec[np.where(r_geq_z)] = np.sqrt(r_top_squared[r_geq_z])
            r_top_vec[np.where(~r_geq_z)] = -np.inf

            if scat[0] <= xcd_right:    # If scatterer is under the element
                # Left:
                left_condition = r_geq_z * (r_top_vec <= scat[0] - xcd_left)
                sir[left_condition] = sir_integ_diff[left_condition]
                # Right:
                right_condition = r_geq_z * (r_top_vec <= xcd_right - scat[0])
                sir[right_condition] += sir_integ_diff[right_condition]
            else:   # If scatterer is not under element (it is at right)
                # Left (only case):
                left_condition = r_geq_z * (r_top_vec <= scat[0] - xcd_left) * (r_top_vec >= scat[0] - xcd_right - 1e-5)
                sir[left_condition] = sir_integ_diff[left_condition]

            ####################### Plot Spatial impulse response
            plt.subplot(2, 2, 2)
            t_plot = (t >= important_times.min()) * (t <= important_times.max())
            plt.plot(t[t_plot] * 1e6, sir[t_plot], linewidth=2)
            # plt.axis([(approx_tof-.75*width/c)*1e6, (approx_tof+.75*width/c)*1e6, -.01*sir.max(), 1.1*sir.max()])
            # plt.axis([(approx_tof - .75 * width / c) * 1e6, (approx_tof + .75 * width / c) * 1e6, 0, .25e7])
            # plt.axis([6, 8, -.2, 2])
            # plt.axis([approx_tof * 1e6 - 1, approx_tof * 1e6 + 1, -.2, 2])
            # plt.axis([t[0] * 1e6, t[-1] * 1e6, -.2, 1.2])
            plt.xlabel('time [μs]')
            plt.grid()
            plt.title('SIR')
            ax = plt.axis()
            plt.axis([(scat_distances[i_dist]-xcd_right-width/4)*1e6/c, (scat_distances[i_dist]-xcd_left+width)*1e6/c, 0, ax[3]*1.1])
            mng = plt.get_current_fig_manager()
            mng.resize(700, 870)

        ####################### SIR Spectrum
        sir_padded = np.concatenate((sir, np.zeros(np.int(sir_size_samples - sir.size))))
        sir_fft = np.fft.fft(sir_padded)
        sir_fft = np.abs(sir_fft)
        sir_fft = np.fft.fftshift(sir_fft)
        frequencies_Hz = np.linspace(-fs / 2, fs / 2, sir_fft.size) / 1e6
        index_freq_monochr = np.int(np.round((1e6 * freq_monochr / (fs / 2)) * np.ceil(sir_fft.size / 2)))
        index_freq_0 = np.int(np.floor(sir_fft.size / 2))
        if monochromatic is True:
            gain_monochr = sir_fft[index_freq_0 + index_freq_monochr]
        else:
            gain_monochr = np.sqrt(np.sum(np.abs(np.power(sir_fft * excit_fft, 2))))

        gains_monochr_angle[i_angle] = gain_monochr
        if anim_angle is True:
            ax_4 = plt.subplot(2, 2, 4)
            plt.plot(frequencies_Hz, sir_fft)
            if monochromatic is True:
                plt.stem([-freq_monochr, freq_monochr],
                         [gain_monochr, gain_monochr], 'r', use_line_collection=True)
                # plt.plot([-2 * fs * 1e-6, 2 * fs * 1e-6], [gain_monochr, gain_monochr], c='r', linewidth=.5)
            else:
                for i_levels in range(indices_lvls.__len__()):
                    ax_4.fill_between(frequencies_Hz[indices_lvls[i_levels]],
                                      0,
                                      # sir_fft[np.where(np.abs(frequencies_Hz) <= 10)].min(),
                                      sir_fft[indices_lvls[i_levels]],
                                      color='red', alpha=1 / num_levels)

                # ax_4.fill_between(frequencies_Hz[indices6dB_left],
                #                  1e8,
                #                  #sir_fft[np.where(np.abs(frequencies_Hz) <= 10)].min(),
                #                  sir_fft[indices6dB_left],
                #                  color='red', alpha=.5)
                # ax_4.fill_between(frequencies_Hz[indices6dB_right],
                #                  1e8,
                #                  #sir_fft[np.where(np.abs(frequencies_Hz) <= 10)].min(),
                #                  sir_fft[indices6dB_right],
                #                  color='red', alpha=.5)
            ax = plt.axis()
            if i_angle == 0:
                max_gain_axis = np.max(sir_fft)
            plt.axis([-10, 10, 0, max_gain_axis*1.05])
            plt.xlabel('Frequency [MHz]')
            plt.title('SIR magnitude spectrum')
            plt.grid()

            plt.subplot(subplot_gainperangle)
            angles_gain_nonzero = np.where(gains_monochr_angle > 0)
            plt.plot(scat_angles[angles_gain_nonzero], gains_monochr_angle[angles_gain_nonzero], c='blue')
            plt.plot(-scat_angles[angles_gain_nonzero], gains_monochr_angle[angles_gain_nonzero], c='blue')
            plt.scatter([scat_angles[i_angle]], [gain_monochr], c='r')
            if i_angle == 0:
                max_monoch_axis = np.max(gains_monochr_angle)
            plt.axis([-95, 95, 0, max_monoch_axis])
            plt.grid()
            plt.xlabel('Angle [degrees]')
            plt.ylabel('Gain')
            plt.title('Directivity')

            plt.savefig('./figs/dist' + "%05d" % (i_dist,) + '_angle' + "%05d" % (i_angle,) + '.png')
        SIR_per_angle.append(sir)

        if anim_time is True:
            makevideo.makevideo('times', frames=15)
    SIR_per_distance.append(SIR_per_angle)

if anim_angle is True:
    makevideo.makevideo('angles')

corr_mtx = np.zeros((scat_distances.size, scat_distances.size, scat_angles.size))
for i_dist in range(scat_distances.size):
    for j_dist in range(scat_distances.size):
        for i_angle in range(scat_angles.size):
            corr_mtx[i_dist, j_dist, i_angle] = \
                correlation(SIR_per_distance[i_dist][i_angle],
                            SIR_per_distance[j_dist][i_angle])

sum_mtx = np.zeros((scat_distances.size, scat_angles.size))
for i_dist in range(scat_distances.size):
    for i_angle in range(scat_angles.size):
        sum_mtx[i_dist][i_angle] = \
            np.sum(SIR_per_distance[i_dist][i_angle])

integral = t * np.sqrt(1 - np.divide(scat[1] ** 2, (c ** 2) * (t ** 2)))
