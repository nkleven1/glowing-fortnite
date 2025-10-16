# This code is authored by Noah Kleven 
# 5/7/25
# This code finds the range of initial conditions for a spacecraft to be captured by the Moon in a 3-body problem.
# The code uses inverse transformation sampling to sample a normal distribution for the initial velocity and angle.


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import random as rd 
from scipy.special import erfinv
import time


# Timer to make sure code is still running
time_start = time.time()

# Constants
G = 6.67259e-20
mu = 3.986e5
mu_m = 4902.8

m1 = 5.974e24
m2 = 7.348e22

t0 = 0
tf = 4 * 24 * 60 * 60  # four days is long enough to fly by the moon

Re = 6378
Rm = 1737
Re_m = 384400
SOI_m_radius = 66100

pi_1 = m1 / (m1 + m2)
pi_2 = m2 / (m1 + m2)

w_m = 2.6653e-6

RE_0 = np.array([-pi_2 * Re_m, 0, 0])
RM_0 = np.array([pi_1 * Re_m, 0, 0])

alt = 200
r0 = Re + alt
v_orbital = np.sqrt(mu / r0)  # circular orbit velocity

# Inverse transform sampling for normal distribution
def inverse_transformation(nrandom_number, gy_inverse): 
    rand_num = []
    for i in range(round(nrandom_number + 1)):
        xi = rd.random()
        rand_num.append(gy_inverse(xi))
    return rand_num

# CFD of normal distribution
mean_y = 0.0 
sigma_y = .5
gy_inverse = lambda x: mean_y + np.sqrt(2.0) * sigma_y * erfinv(2 * x - 1)

###################################
total_sample = 5000  # Number of samples to generate
vel_range = .025 # [km/s] size of velocity range for sampling
phi_range = np.deg2rad(6)  # size of angle range for sampling
v_mag_nominal = 10.9278050
phi_nominal = np.deg2rad(232.20)
##################################


sample = inverse_transformation((total_sample-1), gy_inverse)
sample2 = inverse_transformation((total_sample-1), gy_inverse)
# Storage
min_distance = np.zeros(len(sample))
min_distance_time = np.zeros(len(sample))
total_delta_v = np.full(len(sample), np.nan)

min_total_dv = np.inf
best_index = []
best_v_mag = []
best_phi = []

success_count = 0
fail_count = 0

# Tracking variables for global min and max of closest approaches
min_of_min_distance = np.inf
max_of_min_distance = -np.inf
index_min_min_distance = -1
index_max_min_distance = -1

# Variables to track if ahead or behind the moon
min_ahead = np.inf
max_ahead = -np.inf
index_min_ahead = -1
index_max_ahead = -1

min_behind = np.inf
max_behind = -np.inf
index_min_behind = -1
index_max_behind = -1

# Tracking successful phi and v_mag for scatter plot
successful_phi = []
successful_vmag = []
all_phi = []
all_vmag = []

try:
    for i in range(len(sample)):
        # Make sure the code is still running
        if i % 250 == 0:
            elapsed_time = time.time() - time_start
            print(f"Iteration {i}: Elapsed time: {elapsed_time:.2f} seconds")
            print(f"Percentage of successful trajectories: {success_count / len(sample) * 100:.2f}%")

        v_mag_i = v_mag_nominal + sample[i] * vel_range
        phi_i = phi_nominal + sample2[i] * phi_range
        all_phi.append(np.rad2deg(phi_i))
        all_vmag.append(v_mag_i)

        r0_x = r0 * np.cos(phi_i)
        r0_y = r0 * np.sin(phi_i)
        v_magx = -v_mag_i * np.sin(phi_i)
        v_magy = v_mag_i * np.cos(phi_i)

        r_rel_Earth = np.array([r0_x, r0_y, 0])
        v_rel_Earth = np.array([v_magx, v_magy, 0]) 

        RS_0 = RE_0 + r_rel_Earth
        VS_0 = v_rel_Earth + np.array([0, w_m * r0, 0])

        def rates(t, y):
            R3 = y[:3]
            V = y[3:]
            r31 = np.linalg.norm(R3 - RE_0)
            r32 = np.linalg.norm(R3 - RM_0)

            x_dot2 = (-mu / r31**3) * (R3[0] + pi_2 * Re_m) \
                     - (mu_m / r32**3) * (R3[0] - pi_1 * Re_m) \
                     + (w_m**2) * R3[0] + 2 * w_m * V[1]

            y_dot2 = (-mu / r31**3) * R3[1] \
                     - (mu_m / r32**3) * R3[1] \
                     + (w_m**2) * R3[1] - 2 * w_m * V[0]

            z_dot2 = (-mu / r31**3) * R3[2] \
                     - (mu_m / r32**3) * R3[2]

            return np.concatenate((V, [x_dot2, y_dot2, z_dot2]))

        y0 = np.concatenate((RS_0, VS_0))

        try:
            sol = solve_ivp(rates, [t0, tf], y0, rtol=1e-5, atol=1e-9, method='BDF')
            if not sol.success:
               fail_count += 1
               continue
        except Exception:
            fail_count += 1
            continue

        X, Y, Z = sol.y[0], sol.y[1], sol.y[2]
        Vx, Vy, Vz = sol.y[3], sol.y[4], sol.y[5]
        t = sol.t

        distances = np.linalg.norm(np.vstack((X, Y, Z)).T - RM_0, axis=1)
        min_index = np.argmin(distances)
        min_distance[i] = distances[min_index]
        dist_from_surface = min_distance[i] - Rm

        # Check if the distance from the surface is within the SOI radius
        if dist_from_surface < 0 or dist_from_surface > SOI_m_radius:
            fail_count += 1
            continue

        success_count += 1
        min_distance_time[i] = t[min_index] / 3600
        velocity_at_min = np.linalg.norm([Vx[min_index], Vy[min_index], Vz[min_index]])

        # Delta V calculation
        a_capture = (min_distance[i] + SOI_m_radius) / 2
        dv = abs(velocity_at_min - np.sqrt(mu_m * (2 / min_distance[i] - 1 / a_capture)))
        total_delta_v[i] = dv + v_mag_i - v_orbital

        if total_delta_v[i] < 3.5:
            best_index.append(i)
            best_v_mag.append(v_mag_i)
            best_phi.append(np.rad2deg(phi_i))

        # Update min/max of minimum distances
        if min_distance[i] < min_of_min_distance:
            min_of_min_distance = min_distance[i]
            index_min_min_distance = i
        if min_distance[i] > max_of_min_distance:
            max_of_min_distance = min_distance[i]
            index_max_min_distance = i

        # Save successful case for plotting
        successful_phi.append(np.rad2deg(phi_i))
        successful_vmag.append(v_mag_i)

        # Determine if spacecraft goes ahead or behind the Moon
        # find when the craft crosses the moons x position and use sign of y to determine ahead or behind
        x_diff = np.abs(X - RM_0[0])
        moon_x_cross_index = []

        # Populate moon_x_cross_index
        for j in range(len(x_diff)):
            if x_diff[j] < 200:
                moon_x_cross_index.append(j)

        # Check if moon_x_cross_index is empty
        if not moon_x_cross_index:
            fail_count += 1
            continue

        y_at_moon_x = Y[moon_x_cross_index[0]]

        #debugging
        #print(f"y_at_moon_x: {y_at_moon_x}")

        if y_at_moon_x > 0:  # Ahead
            if min_distance[i] < min_ahead:
               min_ahead = min_distance[i]
               index_min_ahead = i
            if min_distance[i] > max_ahead:
                max_ahead = min_distance[i]
                index_max_ahead = i
        else:  # Behind
            if min_distance[i] < min_behind:
               min_behind = min_distance[i]
               index_min_behind = i
            if min_distance[i] > max_behind:
                max_behind = min_distance[i]
                index_max_behind = i

        # make sure the code is still running
        if i % 50 == 0:
            print(f"iteration {i}/{len(sample)} completed")
         

except Exception as e:
    print(f"Unexpected error at iteration {i}: {e}")
    fail_count += 1

##############################
# Results
#############################
print("\n=== Minimum of Closest Ahead Approaches ===")
if index_min_ahead != -1:
    phi_deg = np.rad2deg(phi_nominal + sample2[index_min_ahead] * phi_range)
    v_mag_val = v_mag_nominal + sample[index_min_ahead] * vel_range
    print(f"Total Delta V for Maneuver: {total_delta_v[index_min_ahead]:.6f} km/s")
    print(f"Min. closest distance to Moon: {min_ahead - Rm:.2f} km")
    print(f"Initial velocity magnitude: {v_mag_val:.6f} km/s")
    print(f"Initial angle (deg): {phi_deg:.6f}")

print("\n=== Maximum of Closest Ahead Approaches ===")
if index_max_ahead != -1:
    phi_deg = np.rad2deg(phi_nominal + sample2[index_max_ahead] * phi_range)
    v_mag_val = v_mag_nominal + sample[index_max_ahead] * vel_range
    print(f"Total Delta V for Maneuver: {total_delta_v[index_max_ahead]:.6f} km/s")
    print(f"Max. closest distance to Moon: {max_ahead - Rm:.2f} km")
    print(f"Initial velocity magnitude: {v_mag_val:.6f} km/s")
    print(f"Initial angle (deg): {phi_deg:.6f}")

print("\n=== Minimum of Closest Behind Approaches ===")
if index_min_behind != -1:
    phi_deg = np.rad2deg(phi_nominal + sample2[index_min_behind] * phi_range)
    v_mag_val = v_mag_nominal + sample[index_min_behind] * vel_range
    print(f"Total Delta V for Maneuver: {total_delta_v[index_min_behind]:.6f} km/s")
    print(f"Min. closest distance to Moon: {min_behind - Rm:.2f} km")
    print(f"Initial velocity magnitude: {v_mag_val:.6f} km/s")
    print(f"Initial angle (deg): {phi_deg:.6f}")

print("\n=== Maximum of Closest Behind Approaches ===")
if index_max_behind != -1:
    phi_deg = np.rad2deg(phi_nominal + sample2[index_max_behind] * phi_range)
    v_mag_val = v_mag_nominal + sample[index_max_behind] * vel_range
    print(f"Total Delta V for Maneuver: {total_delta_v[index_max_behind]:.6f} km/s")
    print(f"Max. closest distance to Moon: {max_behind - Rm:.2f} km")
    print(f"Initial velocity magnitude: {v_mag_val:.6f} km/s")
    print(f"Initial angle (deg): {phi_deg:.6f}")
    
time_end = time.time()
print("\nThe code takes --- %.2f seconds --- to finish" % (time_end - time_start))
print(f"Successful trajectories: {success_count}")
print(f"Percentage of successful trajectories: {success_count / len(sample) * 100:.2f}%")


# Scatter plot
plt.figure()
plt.scatter(all_phi, all_vmag, c='red', alpha=0.5, edgecolors='black', label='Moon-Impact Trajectories')
plt.scatter(successful_phi, successful_vmag, c='green', alpha=0.75, edgecolors='black', label='Successful Trajectories')
plt.scatter(best_phi, best_v_mag, c='blue', alpha=0.75, edgecolors='black', label='Trajectories with Δv < 3.5 km/s')
plt.legend()
plt.xlabel("Initial Angle (deg)")
plt.ylabel("Initial Velocity Magnitude (km/s)")
plt.tight_layout()


# Histogram of min distances
plt.figure()
plt.hist(min_distance, bins=50, color='blue', alpha=0.7, edgecolor='black')
plt.axvline(x=min_of_min_distance, color='red', linestyle='dashed', linewidth=1, label='Min Distance')
plt.axvline(x=max_of_min_distance, color='green', linestyle='dashed', linewidth=1, label='Max Distance')
plt.xlabel("Minimum Distance to Moon (km)")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()

# Histogram of phi values
plt.figure()
plt.hist(all_phi, bins=50, color='blue', alpha=0.7, edgecolor='black')
plt.xlabel("Initial Angle (deg)")
plt.ylabel("Frequency")
plt.tight_layout()

# Histogram of velocity magnitudes
plt.figure()
plt.hist(all_vmag, bins=50, color='blue', alpha=0.7, edgecolor='black')
plt.xlabel("Initial Velocity Magnitude (km/s)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
