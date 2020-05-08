import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import makevideo


class LimitsObjects:
    __x_min = 0
    __x_max = 0
    __y_min = 0
    __y_max = 0
    __z_min = 0
    __z_max = 0
    __axis_x_min = 0
    __axis_x_max = 0
    __axis_y_min = 0
    __axis_y_max = 0
    __axis_z_min = 0
    __axis_z_max = 0

    def set_x_min(self, value):
        self.__x_min = value

    def set_x_max(self, value):
        self.__x_max = value

    def set_y_min(self, value):
        self.__y_min = value

    def set_y_max(self, value):
        self.__y_max = value

    def set_z_min(self, value):
        self.__z_min = value

    def set_z_max(self, value):
        self.__z_max = value

    def calc_axis(self):
        border_ratio = 1.2
        x_center = (self.__x_max + self.__x_min) / 2
        y_center = (self.__y_max + self.__y_min) / 2
        z_center = (self.__z_max + self.__z_min) / 2
        x_length = (self.__x_max - self.__x_min)
        y_length = (self.__y_max - self.__y_min)
        z_length = (self.__z_max - self.__z_min)
        global_length = np.max([x_length, y_length, z_length]) * border_ratio
        self.__axis_x_min = x_center - global_length / 2
        self.__axis_x_max = x_center + global_length / 2
        self.__axis_y_min = y_center - global_length / 2
        self.__axis_y_max = y_center + global_length / 2
        self.__axis_z_min = z_center - global_length / 2
        self.__axis_z_max = z_center + global_length / 2

    def get_axis_xz(self):
        return [self.__axis_x_min, self.__axis_x_max,
                self.__axis_z_min, self.__axis_z_max]

    def get_axis_xy(self):
        return [self.__axis_x_min, self.__axis_x_max,
                self.__axis_y_min, self.__axis_y_max]

###### User configurable parameters ############################
######                              ############################

# Transducer's width [mm]
width = 1.0
# Transducer's length [mm]
length = 10.0
# Scatterer's position (x, z) [mm]
scat = np.array([10, 10])
# Sampling frequency [Hz]
fs = 50e6
# Speed of sound [m/s]
c_m_s = 1485.0
# size (samples) of the zero-padded SIR, both in time and frequency domains
sir_size_samples = 1001
# Exhibit animation along time
anim_time = False
# Exhibit animation along angles
anim_time = False


################################################################
################################################################



# x coordinate of tranducer's left corner [mm]
xcd_left = 0.0
# Speed of sound [mm/s]
c = c_m_s * 1000.0
# x coordinate of transducer's right corner [mm]
xcd_right = xcd_left + width
# y coordinate of transducer's top corner
xdc_top = length/2
# y coordinate of transducer's bottom corner
xdc_bottom = length/2

important_times = list()
important_times.append(np.sqrt((scat[0]-xcd_left)**2 + xdc_top**2 + scat[1]**2))
important_times.append(np.sqrt((scat[0]-xcd_right)**2 + xdc_top**2 + scat[1]**2))
important_times.append(np.sqrt((scat[0]-xcd_left)**2 + scat[1]**2))
important_times.append(np.sqrt((scat[0]-xcd_right)**2 + scat[1]**2))
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
t_hr = np.arange(t_HR_threshold, t_final, 1/fs)
if anim_time:
    t = np.append(t_lr, 2*t_hr[0]-t_hr[1])
    t = np.append(t, t_hr)
else:
    t = t_hr
# Spatial impulse response
sir = np.zeros_like(t)


# Draw transducer's surface
plt.plot([xcd_left, width], [0, 0], linewidth=2)
# Mirror vertical axis
ax = plt.gca()
ax.set_ylim(ax.get_ylim()[::-1])
# Draw scatterer
plt.scatter([scat[0]], [scat[1]])
# Angles array (to draw circle)
angles = np.arange(0, 2*np.pi, .01)
cos_angles = np.cos(angles)
sin_angles = np.sin(angles)

lim_obj = LimitsObjects()
lim_obj.set_x_min(np.min([xcd_left, scat[0]]))
lim_obj.set_x_max(np.max([xcd_right, scat[0]]))
lim_obj.set_y_min(-length/2)
lim_obj.set_y_max(length/2)
lim_obj.set_z_min(0)
lim_obj.set_z_max(scat[1])
lim_obj.calc_axis()


i = 0
for t_now in t:
    arc_total = 0
    # Compute current radius
    r = c * t_now

    # Compute current radius of the projection of the forntwave
    # on the plane z=0.
    # Geometric scheme: https://photos.app.goo.gl/N6HJ6SSz3kSYaYiH9
    if r >= scat[1]:
        z_proj_exists = True
        beta = np.arccos(scat[1]/r)
        r_top = r*np.sin(beta)
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
    plt.clf()

    ####################### Front view
    plt.subplot(2, 2, 1)
    # Draw line z=0
    plt.plot([-1e10, 1e10], [0, 0], c='gray', linewidth=.5)
    # Draw transducer's surface
    plt.plot([xcd_left, width], [0, 0], linewidth=2)
    # Draw scatterer
    plt.scatter([scat[0]], [scat[1]])
    # Draw circle
    circle_x = scat[0] + r * cos_angles
    circle_z = scat[1] + r * sin_angles
    plt.plot(circle_x, circle_z, 'k')
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
    #if intersection_1 == True:
    #    plt.scatter([x_1], [0], c='r')
    #if intersection_2 == True:
    #    plt.scatter([x_2], [0], c='r')

    ####################### Top view
    plt.subplot(2, 2, 3)
    # Draw transducer's surface
    xdc_rect = patches.Rectangle([xcd_left, -length/2], width=width, height=length, fill=True, color='blue')
    ax = plt.gca()
    ax.add_patch(xdc_rect)
    boundaries_angles = list()
    # Draw circle
    if z_proj_exists:
        circle_x = scat[0] + r_top * cos_angles
        circle_y = r_top * sin_angles
        plt.plot(circle_x, circle_y, 'k')

        intercept_left = False
        y_SQ = -xcd_left**2 + 2*scat[0]*xcd_left - scat[0]**2 + r_top**2
        if 0 <= y_SQ <= (length/2)**2:
            intercept_left = True
            y = np.sqrt(y_SQ)
            plt.scatter([xcd_left, xcd_left], [y, -y], c='r')
            if np.abs(xcd_left-scat[0]) < 1e-60:
                angle = np.pi/2
            else:
                angle = np.arctan(np.divide(y, (xcd_left-scat[0])))
            if angle < 0:
                angle += np.pi
            boundaries_angles.append(angle)
        intercept_right = False
        y_SQ = -xcd_right ** 2 + 2 * scat[0] * xcd_right - scat[0] ** 2 + r_top ** 2
        if 0 <= y_SQ <= (length/2)**2:
            intercept_right = True
            y = np.sqrt(y_SQ)
            plt.scatter([xcd_right, xcd_right], [y, -y], c='r')
            angle = np.arctan(np.divide(y, (xcd_right-scat[0])))
            if angle < 0:
                angle += np.pi
            boundaries_angles.append(angle)
        intercept_top_left = False
        intercept_top_right = False
        a = 1
        b = -2*scat[0]
        C = scat[0]**2 + xdc_top**2 - r_top**2
        delta = b**2 - 4*a*C
        if delta >= 0:
            x_top_left = (-b + np.sqrt(delta))/(2*a)
            if xcd_left <= x_top_left <= xcd_right:
                intercept_top_left = True
                plt.scatter([x_top_left, x_top_left], [length/2, -length/2], c='r')
                angle = np.arctan((length/2) / (x_top_left-scat[0]))
                if angle < 0:
                    angle += np.pi
                boundaries_angles.append(angle)
            x_top_right = (-b - np.sqrt(delta)) / (2 * a)
            if xcd_left <= x_top_right <= xcd_right:
                intercept_top_right = True
                plt.scatter([x_top_right, x_top_right], [length / 2, -length / 2], c='r')
                angle = np.arctan((length / 2) / (x_top_right-scat[0]))
                if angle < 0:
                    angle += np.pi
                boundaries_angles.append(angle)
        if xcd_left <= scat[0]+r_top <= xcd_right: # If the right-most point of top circle is in the transducer area...
            # ...the the first interval *belongs* to the active arcs
            boundaries_angles.append(0)
        boundaries_angles.sort()
        # Draw arcs
        circle_x = scat[0] + r_top * cos_angles
        circle_y = r_top * sin_angles
        idx_bound = 0
        while idx_bound < boundaries_angles.__len__():
            if idx_bound+1 < boundaries_angles.__len__():
                arc_indices = np.multiply(angles >= boundaries_angles[idx_bound],
                                          angles <= boundaries_angles[idx_bound+1])
                arc_total += 2*np.abs(boundaries_angles[idx_bound+1]-boundaries_angles[idx_bound])
            else:
                arc_indices = np.multiply(angles >= boundaries_angles[idx_bound],
                                          angles <= np.pi)
                arc_total += 2 * np.abs(np.pi - boundaries_angles[idx_bound])
            plt.plot(circle_x[arc_indices], circle_y[arc_indices], c='r')
            plt.plot(circle_x[arc_indices], -circle_y[arc_indices], c='r')
            idx_bound += 2
    plt.title('Top view (z=0)')
    plt.xlabel('x[mm]')
    plt.ylabel('y[mm]')
    plt.axis('square')
    plt.axis(lim_obj.get_axis_xy())
    sir[i] = arc_total

    ####################### Spatial impulse response
    plt.subplot(2, 2, 2)
    plt.plot(t[:i] * 1e6, sir[:i])
    #plt.axis([t[0] * 1e6, t[-1] * 1e6, -.2, 1.2])
    plt.xlabel('time [μs] (current: ' + "%.2f" % (t_now * 1e6,) + ' μs)')
    plt.grid()
    plt.title('Spatial Impulse Response')
    plt.axis('square')

    mng = plt.get_current_fig_manager()
    mng.resize(700, 800)
    i += 1
    plt.savefig('./figs/' + "%05d" % (i,) + '.png')

####################### SIR Spectrum
sir_padded = np.concatenate((sir, np.zeros(sir_size_samples - sir.size)))
plt.subplot(2,2,4)
sir_fft = np.fft.fft(sir_padded)
sir_fft = np.abs(sir_fft)
sir_fft = np.fft.fftshift(sir_fft)
frequencies_Hz = np.linspace(-fs/2, fs/2, sir_fft.size) /1e6
plt.semilogy(frequencies_Hz, sir_fft)
plt.grid()
plt.xlabel('Frequency [MHz]')
plt.ylabel('Amplitude')
plt.title('SIR magnitude spectrum')

makevideo.makevideo()
