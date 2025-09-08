"""
2D particle gas in a box — cleaned, optimized and modeled version.

Features:
- Hard-sphere elastic collisions between equal-mass particles
- Wall reflections with impulse accumulation -> pressure estimate
- KDTree neighbor search for collision detection
- Kinetic energy histogram and temperature estimate (2D equipartition)
- Simple animation (matplotlib) showing particle positions

Author: Alejandro (adapted & improved)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import time

# ------------------------
# Parameters (tunable)
# ------------------------
W, H = 10.0, 10.0         # box width and height
N = 100                   # number of particles
r = 0.12                  # particle radius (collision distance ~ 2*r)
m = 1.0                   # particle mass (all equal)
kB = 1.380649e-23         # Boltzmann constant (SI) - used only for scaling if desired
dt = 0.02                 # time step
steps = 2000              # number of time steps to simulate
plot_every = 5            # update animation every n steps
seed = 42                 # RNG seed for reproducibility

np.random.seed(seed)

# ------------------------
# Initialization
# ------------------------
# Positions uniformly in box avoiding edges by radius
pos = np.empty((N, 2))
pos[:, 0] = np.random.uniform(r, W - r, size=N)
pos[:, 1] = np.random.uniform(r, H - r, size=N)

# Small random velocities
v_scale = 0.5
angles = np.random.uniform(0, 2 * np.pi, size=N)
speeds = np.random.normal(loc=0.6 * v_scale, scale=0.15 * v_scale, size=N)
vel = np.zeros((N, 2))
vel[:, 0] = speeds * np.cos(angles)
vel[:, 1] = speeds * np.sin(angles)

# Diagnostics storage
impulse_x_total = 0.0   # cumulative impulse delivered to vertical walls
impulse_y_total = 0.0   # cumulative impulse delivered to horizontal walls
pressure_history = []
time_history = []
kinetic_history = []

# Helper: resolve elastic collision for equal masses
def resolve_pair_collision(i, j, pos, vel):
    """Resolve elastic collision between particles i and j (equal mass)."""
    ri = pos[i]; rj = pos[j]
    vi = vel[i]; vj = vel[j]
    delta = ri - rj
    dist = np.linalg.norm(delta)
    if dist == 0:
        # numerical safety: jitter
        delta = np.random.randn(2) * 1e-6
        dist = np.linalg.norm(delta)
    n = delta / dist  # unit normal from j->i
    # relative velocity along normal
    rel = np.dot(vi - vj, n)
    if rel >= 0:
        # moving apart or tangential -> no normal impulse
        return
    # For equal masses, swapping normal components is equivalent:
    vi_n = np.dot(vi, n)
    vj_n = np.dot(vj, n)
    vi_t = vi - vi_n * n
    vj_t = vj - vj_n * n
    vel[i] = vi_t + vj_n * n
    vel[j] = vj_t + vi_n * n
    # Separate slightly to avoid overlap
    overlap = 2 * r - dist
    if overlap > 0:
        shift = 0.5 * overlap + 1e-6
        pos[i] += n * shift
        pos[j] -= n * shift

# ------------------------
# Simulation loop
# ------------------------
fig, ax = plt.subplots(figsize=(6, 6))
scat = ax.scatter(pos[:, 0], pos[:, 1], s=50)
ax.set_xlim(0, W); ax.set_ylim(0, H)
ax.set_aspect('equal')
ax.set_title('2D gas in a box')
plt.ion()
plt.show()

start_time = time.time()
sim_time = 0.0

for step in range(steps):
    # integrate positions
    pos += vel * dt
    sim_time += dt

    # wall collisions & accumulate impulses
    # Left wall (x < r)
    left_hits = pos[:, 0] < r
    if np.any(left_hits):
        # impulse = 2 m v_x for particles reflecting (only those moving left)
        vx = vel[left_hits, 0]
        # Only count those actually moving toward the wall
        incoming = vx < 0
        impulse_x_total += np.sum(2.0 * m * np.abs(vx[incoming]))
        # reflect velocities
        vel[left_hits, 0] *= -1
        # reposition inside
        pos[left_hits, 0] = r + (r - pos[left_hits, 0])

    # Right wall (x > W - r)
    right_hits = pos[:, 0] > (W - r)
    if np.any(right_hits):
        vx = vel[right_hits, 0]
        incoming = vx > 0
        impulse_x_total += np.sum(2.0 * m * np.abs(vx[incoming]))
        vel[right_hits, 0] *= -1
        pos[right_hits, 0] = (W - r) - (pos[right_hits, 0] - (W - r))

    # Bottom wall (y < r)
    bottom_hits = pos[:, 1] < r
    if np.any(bottom_hits):
        vy = vel[bottom_hits, 1]
        incoming = vy < 0
        impulse_y_total += np.sum(2.0 * m * np.abs(vy[incoming]))
        vel[bottom_hits, 1] *= -1
        pos[bottom_hits, 1] = r + (r - pos[bottom_hits, 1])

    # Top wall (y > H - r)
    top_hits = pos[:, 1] > (H - r)
    if np.any(top_hits):
        vy = vel[top_hits, 1]
        incoming = vy > 0
        impulse_y_total += np.sum(2.0 * m * np.abs(vy[incoming]))
        vel[top_hits, 1] *= -1
        pos[top_hits, 1] = (H - r) - (pos[top_hits, 1] - (H - r))

    # Particle-particle collisions using KDTree (find pairs within 2r)
    tree = cKDTree(pos)
    pairs = tree.query_pairs(r=2 * r)
    if pairs:
        for (i, j) in pairs:
            # double-check separation and handle collision
            delta = pos[i] - pos[j]
            dist = np.linalg.norm(delta)
            if dist < 2 * r - 1e-12:
                resolve_pair_collision(i, j, pos, vel)

    # Diagnostics every few steps
    if step % plot_every == 0:
        scat.set_offsets(pos)
        plt.pause(0.001)

    # store kinetic energy and pressure estimate
    speeds2 = np.sum(vel**2, axis=1)
    kinetic = 0.5 * m * np.mean(speeds2)      # mean kinetic energy per particle
    kinetic_history.append(kinetic)

    # pressure estimate: total impulse per unit time divided by box area
    # Using impulses accumulated so far and elapsed simulation time
    if sim_time > 0:
        total_impulse = impulse_x_total + impulse_y_total
        pressure_est = total_impulse / sim_time / (W * H)  # N / m^2 (approx)
    else:
        pressure_est = 0.0
    pressure_history.append(pressure_est)
    time_history.append(sim_time)

end_time = time.time()
print(f"Simulation done in {end_time - start_time:.2f} s")

plt.ioff()
plt.close(fig)

# ------------------------
# Post-processing & plots
# ------------------------
# Kinetic energy histogram (per-particle kinetic energies)
per_particle_kin = 0.5 * m * np.sum(vel**2, axis=1)
plt.figure(figsize=(6, 4))
plt.hist(per_particle_kin, bins=25, color='C0', edgecolor='k', alpha=0.7)
plt.xlabel('Kinetic energy (J)')
plt.ylabel('Number of particles')
plt.title('Kinetic energy distribution (final)')
plt.grid(alpha=0.3)
plt.show()

# Temperature estimate using equipartition (2D): <E_kin> = kB * T
T_est = np.mean(per_particle_kin) / kB
print(f"Estimated temperature (2D equipartition): T = {T_est:.3e} K")

# Pressure vs time
plt.figure(figsize=(6, 4))
plt.plot(time_history, pressure_history, color='C1')
plt.xlabel('Time (s)')
plt.ylabel('Estimated Pressure (Pa)')
plt.title('Pressure estimate vs time')
plt.grid(alpha=0.3)
plt.show()

# Average pressure and energy
print(f"Final average pressure (Pa): {np.mean(pressure_history[-100:]):.3e}")
print(f"Final mean kinetic energy per particle (J): {np.mean(per_particle_kin):.3e}")


