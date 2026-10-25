import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
import sys


def x_position(t):
    """Nozzle position at physical time t [s], independent of integration dt."""
    return x_c + a * np.cos(Omega * t)


def y_position(t):
    return y_c - b * np.sin(Omega * t)

# Material properties and heat-transfer coefficients in SI units
C_to_Kelvin = 273.15
lam = 0.177
C_p = 2080
rho = 1050
alpha_air = 25 #25
alpha_bed = 50  # Assumed effective contact coefficient [W/(m^2 K)]
height = 2e-4
Te = 300
Tb = 105 + C_to_Kelvin
Tn = 240 + C_to_Kelvin

a = 0.1
b = 0.07
dx = dy = 0.29
x_c = dx / 2
y_c = dy / 2
printing_time = 20.0
simulation_time = 40.0  # Includes cooling after printing stops
Omega = 2 * np.pi / printing_time
delta_s_target = 0.5e-3
uniformity_tolerance = 1.0  # K, assessed after closing the ring

a_th = lam / (rho * C_p)
B_air = 3 * alpha_air / (rho * C_p * height)
B_bed = alpha_bed / (rho * C_p * height)

# First calculate geometry, without using the thermal time step.
t_geometry = np.linspace(0, printing_time, 10001)
x_geometry = x_position(t_geometry)
y_geometry = y_position(t_geometry)
s_geometry = np.r_[0.0, np.cumsum(np.hypot(
    np.diff(x_geometry), np.diff(y_geometry)
))]
length = s_geometry[-1]
n_segments = int(np.ceil(length / delta_s_target))
delta_s = length / n_segments
s_edges = np.linspace(0, length, n_segments + 1)
s_centres = (s_edges[:-1] + s_edges[1:]) / 2
x_segments = np.interp(s_centres, s_geometry, x_geometry)
y_segments = np.interp(s_centres, s_geometry, y_geometry)

# Cellwise deposition: activate a complete cell at its downstream edge.
# Refine delta_s to reduce this approximation to continuous deposition.
deposition_times = np.interp(s_edges[1:], s_geometry, t_geometry)

# Forward Euler stability bound with a safety factor.
dt_limit = 1 / (2 * a_th / delta_s**2 + B_air + B_bed)
n_steps = int(np.ceil(simulation_time / (dt_limit / 10)))
# Include every deposition event exactly, splitting time steps as necessary.
time = np.unique(np.r_[
    np.linspace(0, simulation_time, n_steps + 1), deposition_times
])
n_steps = len(time) - 1

# Columns are physical cells; rows are time instants. NaN = not yet printed.
T_history = np.full((len(time), n_segments), np.nan)
Temp = np.empty(0)
next_segment = 0

for n in range(n_steps):
    dt = time[n + 1] - time[n]
    ring_closed = time[n] >= printing_time

    if len(Temp):
        conduction = np.zeros_like(Temp)
        if ring_closed:
            # Periodic neighbours after the entire path has been deposited.
            conduction = a_th * (
                np.roll(Temp, 1) - 2 * Temp + np.roll(Temp, -1)
            ) / delta_s**2
        else:
            # Pairwise exchange; no heat flux through the two free ends.
            exchange = a_th * np.diff(Temp) / delta_s**2
            conduction[:-1] += exchange
            conduction[1:] -= exchange

        dot_T = (conduction - B_air * (Temp - Te)
                 + B_bed * (Tb - Temp))
        Temp = Temp + dt * dot_T

    # New material starts at Tn once, after existing material has advanced.
    while (next_segment < n_segments
           and deposition_times[next_segment] <= time[n + 1]):
        Temp = np.append(Temp, Tn)
        next_segment += 1

    T_history[n + 1, :len(Temp)] = Temp

T_equilibrium = (3 * alpha_air * Te + alpha_bed * Tb) / (
    3 * alpha_air + alpha_bed
)
completed = np.flatnonzero(time >= printing_time)
spread = np.ptp(T_history[completed], axis=1)
uniform = completed[spread < uniformity_tolerance]
print(f'Path length: {length:.4f} m')
print(f'Segments: {n_segments}; delta_s: {delta_s * 1000:.4f} mm')
print(f'Maximum integration step: {np.diff(time).max():.5f} s')
print(f'Expected equilibrium: {T_equilibrium - C_to_Kelvin:.2f} deg C')
if len(uniform):
    print(f'First near-uniform state after closure: t = {time[uniform[0]]:.2f} s')
else:
    print('Not yet near-uniform; increase simulation_time.')

# Animate stored results; display frames are independent of solver steps.
fps = 15
playback_speed = 2.0  # Simulated seconds per playback second
frame_times = np.unique(np.r_[
    np.arange(0, simulation_time, playback_speed / fps),
    printing_time, simulation_time,
])
fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
ax.plot(x_geometry, y_geometry, color='0.85', linewidth=1, zorder=0)
# Fixed nonlinear colour scale: expand differences near thermal equilibrium.
# Only the display changes; all simulated temperatures remain unchanged.
temperature_norm = PowerNorm(
    gamma=1,
    vmin=T_equilibrium - C_to_Kelvin,
    vmax=Tn - C_to_Kelvin,
    clip=True,
)
points = ax.scatter(
    [], [], c=np.empty(0), cmap='inferno',
    norm=temperature_norm, s=14,
)
nozzle, = ax.plot([], [], 'o', color='deepskyblue',
                  markeredgecolor='black', markersize=7, label='Nozzle')
status = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top',
                 bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'))
ax.set(xlabel='x position [m]', ylabel='y position [m]',
       xlim=(0, dx), ylim=(0, dy))
ax.set_aspect('equal')
ax.grid(True)
ax.legend(loc='lower right')
fig.colorbar(points, ax=ax,
             label='Temperature [deg C] (nonlinear colour scale)')


def update_animation(display_time):
    # Show the most recent completed integration state (never a future state).
    frame = np.searchsorted(time, display_time, side='right') - 1
    temperatures = T_history[frame]
    active = np.isfinite(temperatures)
    points.set_offsets(np.column_stack((x_segments[active], y_segments[active])))
    points.set_array(temperatures[active] - C_to_Kelvin)
    printing = display_time < printing_time
    nozzle.set_visible(printing)
    if printing:
        nozzle.set_data([x_position(display_time)], [y_position(display_time)])
    phase = 'Printing' if printing else 'Cooling: deposition finished'
    ax.set_title(f'Filament temperature | t = {display_time:.2f} s')
    status.set_text(f'{phase}\nDeposited cells: {active.sum()} / {n_segments}')
    return points, nozzle, status


# Keep a reference so the animation is not garbage-collected before display.
animation = FuncAnimation(fig, update_animation, frames=frame_times,
                          interval=1000 / fps, blit=False, repeat=True)

# Optional portable export: python Q1_6.py --save-gif
if '--save-gif' in sys.argv:
    output = Path(__file__).resolve().parent / 'output' / 'Q1_6_temperature.gif'
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output, writer=PillowWriter(fps=fps), dpi=90)
    print(f'Animation saved to: {output}')

# Optional PNG sequence for LaTeX: python Q1_6.py --save-frames
if '--save-frames' in sys.argv:
    frames_directory = Path(__file__).resolve().parent / 'frames'
    frames_directory.mkdir(parents=True, exist_ok=True)
    export_frame_count = 30
    export_times = np.linspace(0, simulation_time, export_frame_count)
    export_fps = export_frame_count / (simulation_time / playback_speed)
    # Remove only obsolete numbered frames produced by this exporter.
    for old_frame in frames_directory.glob('frame-*.png'):
        if old_frame.stem.removeprefix('frame-').isdigit():
            old_frame.unlink()
    for frame_number, display_time in enumerate(export_times):
        update_animation(display_time)
        fig.savefig(frames_directory / f'frame-{frame_number:03d}.png', dpi=90)
    # Store matching frame bounds rather than assuming the duration is fixed.
    latex_command = (
        r'\animategraphics[controls,loop,width=0.8\linewidth]'
        f'{{{export_fps:g}}}{{frames/frame-}}{{000}}{{{export_frame_count - 1:03d}}}'
    )
    (frames_directory / 'animation.tex').write_text(
        '% Requires \\usepackage{animate} in the preamble.\n'
        + latex_command + '\n', encoding='utf-8'
    )
    print(f'Exported {export_frame_count} PNG frames to: {frames_directory}')
    print(latex_command)
    update_animation(frame_times[0])

# Separate diagnostic plot; the animation and its GIF export stay unchanged.
fig_ends, ax_ends = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
ax_ends.plot(time, T_history[:, 0] - C_to_Kelvin,
             label='First segment', color='tab:blue')
ax_ends.plot(time, T_history[:, -1] - C_to_Kelvin,
             label='Last segment', color='tab:orange')
ax_ends.axvline(printing_time, color='black', linestyle='--',
                label='Ring closure')
equilibrium_celsius = T_equilibrium - C_to_Kelvin
ax_ends.axhline(equilibrium_celsius, color='tab:red', linestyle=':',
                linewidth=1.5, label='_nolegend_', zorder=4)
ax_ends.annotate(
    rf'$T_\infty = {equilibrium_celsius:.2f}\,^\circ\mathrm{{C}}$',
    xy=(0.03, equilibrium_celsius), xycoords=('axes fraction', 'data'),
    xytext=(0, 7), textcoords='offset points',
    color='tab:red', ha='left', va='bottom',
    bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=2),
)
# Make the last segment's initial value visible at its deposition instant.
ax_ends.scatter([printing_time], [Tn - C_to_Kelvin],
                color='tab:orange', s=25, zorder=3)
ax_ends.set(xlabel='Time [s]', ylabel='Temperature [deg C]',
            title='Temperature of the first and last filament segments',
            xlim=(0, simulation_time))
ax_ends.grid(True)
ax_ends.legend()
ends_output = Path(__file__).resolve().parent / 'output' / 'Q1_6_end_segments.png'
ends_output.parent.mkdir(parents=True, exist_ok=True)
fig_ends.savefig(ends_output, dpi=180)
print(f'Segment temperature plot saved to: {ends_output}')

plt.show()
