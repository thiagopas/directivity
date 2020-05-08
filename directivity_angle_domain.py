import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import makevideo
import geometry

###### User configurable parameters ############################
######                              ############################

# Transducer's width [mm]
width = 5.0
# Transducer's length [mm]
length = 10.0
# Sampling frequency [Hz]
fs = 500e6
# Speed of sound [m/s]
c_m_s = 1485.0
# size (samples) of the zero-padded SIR, both in time and frequency domains
sir_size_samples = 1001
# Exhibit animation along time
anim_time = True
# Exhibit animation along angles
anim_angle = False
# Scatterer angles (degrees)
scat_angles = np.linspace(0, 90, 1)
# Distance from scatterer to transducer center
scat_distances = np.linspace(10, 600, 1)
# Frequency (monochromatic excitation) [MHz]
freq_monochr = 5.
# Gain (at freq_monochr) per angle
gains_monochr_angle = np.zeros_like(scat_angles)


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
# y coordinate of transducer's top corner
xdc_top = length/2
# y coordinate of transducer's bottom corner
xdc_bottom = length/2
# Wavelength [mm]
wavelen = c/(freq_monochr*1e6)
# Near field distance (Fraunhofer distance) [mm]
fraunhofer_d = np.pi * (np.max([width, length])/2)**2 / wavelen

makevideo.delfigs()

subplots = list()
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
        important_times.append(np.sqrt((scat[0] - xcd_left) ** 2 + xdc_top ** 2 + scat[1] ** 2))
        important_times.append(np.sqrt((scat[0] - xcd_right) ** 2 + xdc_top ** 2 + scat[1] ** 2))
        important_times.append(np.sqrt((scat[0] - xcd_left) ** 2 + scat[1] ** 2))
        important_times.append(np.sqrt((scat[0] - xcd_right) ** 2 + scat[1] ** 2))
        if xcd_left <= scat[0] <= xcd_right:
            important_times.append(scat[1])
        important_times = np.array(important_times)
        important_times = important_times / c
        # Simulation initial time [s]
        t_init = 0.0
        # Simulation final time
        if anim_time:
            t_final = important_times.max() + 1e-6
        else:
            t_final = important_times.max()
        # Simulation toggle time resolution
        t_HR_threshold = important_times.min()

        # Time array
        t_lr = np.arange(t_init, t_HR_threshold, 1e-6)
        t_hr = np.arange(t_HR_threshold, t_final, 1 / fs)
        if anim_time:
            t = np.append([], 2 * t_hr[0] - t_hr[1])
            t = np.append(t, t_hr)
        else:
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
        lim_obj.set_y_min(-length / 2)
        lim_obj.set_y_max(length / 2)
        lim_obj.set_z_min(0)
        lim_obj.set_z_max(scat[1])
        lim_obj.calc_axis()

        i = 0
        for t_now in t:
            arc_total = 0
            # Compute current radius
            r = c * t_now

            # Compute current radius of the projection of the frontwave
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
                subplots.append(plt.subplot(2, 2, 1))
                # Draw line z=0
                plt.plot([-1e10, 1e10], [0, 0], c='gray', linewidth=.5)
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
                plt.title('Front view (y=0)')
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
            subplots.append(plt.subplot(2, 2, 3))
            if anim_time is True:
                # Draw transducer's surface
                xdc_rect = patches.Rectangle([xcd_left, -length / 2], width=width, height=length, fill=True,
                                             color='blue')
                # Draw scatterer
                plt.scatter([scat[0]], [0], c='k')
                ax = plt.gca()
                ax.add_patch(xdc_rect)
            boundaries_angles = list()
            # Draw circle
            if z_proj_exists:
                circle_x = scat[0] + r_top * cos_angles
                circle_y = r_top * sin_angles
                if anim_time is True:
                    plt.plot(circle_x, circle_y, 'k')

                intercept_left = False
                y_SQ = -xcd_left ** 2 + 2 * scat[0] * xcd_left - scat[0] ** 2 + r_top ** 2
                if 0 <= y_SQ <= (length / 2) ** 2:
                    intercept_left = True
                    y = np.sqrt(y_SQ)
                    if anim_time is True:
                        plt.scatter([xcd_left, xcd_left], [y, -y], c='r')
                    if np.abs(xcd_left - scat[0]) < 1e-60:
                        angle = np.pi / 2
                    else:
                        angle = np.arctan(np.divide(y, (xcd_left - scat[0])))
                    if angle < 0:
                        angle += np.pi
                    boundaries_angles.append(angle)
                intercept_right = False
                y_SQ = -xcd_right ** 2 + 2 * scat[0] * xcd_right - scat[0] ** 2 + r_top ** 2
                if 0 <= y_SQ <= (length / 2) ** 2:
                    intercept_right = True
                    y = np.sqrt(y_SQ)
                    if anim_time is True:
                        plt.scatter([xcd_right, xcd_right], [y, -y], c='r')
                    angle = np.arctan(np.divide(y, (xcd_right - scat[0])))
                    if angle < 0:
                        angle += np.pi
                    boundaries_angles.append(angle)
                intercept_top_left = False
                intercept_top_right = False
                a = 1
                b = -2 * scat[0]
                C = scat[0] ** 2 + xdc_top ** 2 - r_top ** 2
                delta = b ** 2 - 4 * a * C
                if delta >= 0:
                    x_top_left = (-b + np.sqrt(delta)) / (2 * a)
                    if xcd_left <= x_top_left <= xcd_right:
                        intercept_top_left = True
                        if anim_time is True:
                            plt.scatter([x_top_left, x_top_left], [length / 2, -length / 2], c='r')
                        angle = np.arctan((length / 2) / (x_top_left - scat[0]))
                        if angle < 0:
                            angle += np.pi
                        boundaries_angles.append(angle)
                    x_top_right = (-b - np.sqrt(delta)) / (2 * a)
                    if xcd_left <= x_top_right <= xcd_right:
                        intercept_top_right = True
                        if anim_time is True:
                            plt.scatter([x_top_right, x_top_right], [length / 2, -length / 2], c='r')
                        angle = np.arctan((length / 2) / (x_top_right - scat[0]))
                        if angle < 0:
                            angle += np.pi
                        boundaries_angles.append(angle)
                if xcd_left <= scat[
                    0] + r_top <= xcd_right:  # If the right-most point of top circle is in the transducer area...
                    # ...the the first interval *belongs* to the active arcs
                    boundaries_angles.append(0)
                boundaries_angles.sort()
                # Draw arcs
                circle_x = scat[0] + r_top * cos_angles
                circle_y = r_top * sin_angles
                idx_bound = 0
                #if boundaries_angles.__len__() == 1:
                    #if boundaries_angles[0] == 0:
                        #boundaries_angles.clear()
                while idx_bound < boundaries_angles.__len__():
                    if idx_bound + 1 < boundaries_angles.__len__():
                        arc_indices = np.multiply(angles >= boundaries_angles[idx_bound],
                                                  angles <= boundaries_angles[idx_bound + 1])
                        arc_total += 2 * np.abs(boundaries_angles[idx_bound + 1] - boundaries_angles[idx_bound])
                    else:
                        arc_indices = np.multiply(angles >= boundaries_angles[idx_bound],
                                                  angles <= np.pi)
                        arc_total += 2 * np.abs(np.pi - boundaries_angles[idx_bound])
                    if anim_time is True:
                        plt.plot(circle_x[arc_indices], circle_y[arc_indices], c='r')
                        plt.plot(circle_x[arc_indices], -circle_y[arc_indices], c='r')
                    idx_bound += 2
            if anim_time is True:
                plt.title('Top view (z=0)')
                plt.xlabel('x[mm]')
                plt.ylabel('y[mm]')
                plt.axis('square')
                plt.axis(lim_obj.get_axis_xy())
            sir[i] = arc_total / 2 * np.pi

            ####################### Spatial impulse response
            if i == t.size - 1 or anim_time is True:
                subplots.append(plt.subplot(2, 2, 2))
                plt.plot(t[:i] * 1e6, sir[:i])
                # plt.axis([6, 8, -.2, 2])
                ax = plt.axis()
                plt.axis([approx_tof * 1e6 - 1, approx_tof * 1e6 + 1, -.2, ax[3]])
                # plt.axis([t[0] * 1e6, t[-1] * 1e6, -.2, 1.2])
                plt.xlabel('time [μs] (current: ' + "%.2f" % (t_now * 1e6,) + ' μs)')
                plt.grid()
                plt.title('Spatial Impulse Response')

            if i == t.size - 1 or anim_time is True:
                mng = plt.get_current_fig_manager()
                mng.resize(700, 870)
            if anim_time is True:
                plt.savefig('./figs/' + "%05d" % (i,) + '.png')

            i += 1

        ####################### SIR Spectrum
        sir_padded = np.concatenate((sir, np.zeros(sir_size_samples - sir.size)))
        sir_fft = np.fft.fft(sir_padded)
        sir_fft = np.abs(sir_fft)
        sir_fft = np.fft.fftshift(sir_fft)
        frequencies_Hz = np.linspace(-fs / 2, fs / 2, sir_fft.size) / 1e6
        index_freq_monochr = np.int(np.round((1e6 * freq_monochr / (fs / 2)) * np.ceil(sir_fft.size / 2)))
        index_freq_0 = np.int(np.floor(sir_fft.size / 2))
        gain_monochr = sir_fft[index_freq_0 + index_freq_monochr]
        gains_monochr_angle[i_angle] = gain_monochr
        if anim_angle is True:
            subplots.append(plt.subplot(2, 2, 4))
            plt.semilogy(frequencies_Hz, sir_fft)
            plt.scatter([-freq_monochr, freq_monochr],
                        [gain_monochr, gain_monochr], c='r')
            plt.plot([-2 * fs * 1e-6, 2 * fs * 1e-6], [gain_monochr, gain_monochr], c='r', linewidth=.5)
            plt.axis([-10, 10, 1e-2, 1e2])
            plt.xlabel('Frequency [MHz]')
            plt.title('SIR magnitude spectrum')

            plt.subplot(2, 2, 3)
            angles_gain_nonzero = np.where(gains_monochr_angle > 0)
            plt.plot(scat_angles[angles_gain_nonzero], gains_monochr_angle[angles_gain_nonzero], c='blue')
            plt.plot(-scat_angles[angles_gain_nonzero], gains_monochr_angle[angles_gain_nonzero], c='blue')
            plt.scatter([scat_angles[i_angle]], [gain_monochr], c='r')
            plt.axis([-95, 95, -.5, 8])
            plt.grid()
            plt.xlabel('Angle [degrees]')
            plt.ylabel('Gain')

            plt.savefig('./figs/dist' + "%05d" % (i_dist,) + '_angle' + "%05d" % (i_angle,) + '.png')
        SIR_per_angle.append(sir)

        if anim_time is True:
            makevideo.makevideo('times')
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
