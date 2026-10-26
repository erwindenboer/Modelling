import control as co
from scipy.integrate import solve_ivp
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, FancyArrowPatch
from pathlib import Path
import sys
from enum import Enum, auto

class Mode(Enum):
    PRINTING = auto()
    MOVING_TO_WIPE = auto()
    WIPING = auto()
    LOADING = auto()
    HEATING = auto()
    RETURNING = auto()
    PARKING = auto()
    WAITING = auto()
    FINISHED = WAITING  # Compatibility name for the separate Q3.2 example.

# System parameters
r_m         = 11e-3 # motor radius
K_m         = 0.07  # motor constant
L_m         = 10e-3 # motor inductance
m           = 60e-3 # printer head mass
J_m         = 8e-6  # motor inertia
R_m         = 90    # motor resistance
mu_lx       = 0.002 # linear bearing friction, x-direction
mu_ly       = 0.001 # linear bearing friction, y-direction
mu_r        = 0.003 # Rolling bearing friction
dx          = 0.29  # Print bed length
dy          = 0.29  # Print bed height

T_e         = 300
T_ref       = 513.15 # 240 degrees celsius

R_tn        = 9
R_n         = 8
radius      = 6e-3
l_nozzle    = 12.5e-3
rho_brass = 8530
cp_brass = 377
L_n = 10e-3
C_n = 100e-6
u_n_eq      = np.sqrt((R_n/R_tn)*(T_ref-T_e))
volume_nozzle = np.pi * radius**2 * l_nozzle
mass_nozzle = rho_brass * volume_nozzle
C_tn = mass_nozzle * cp_brass

# Print geometry: change to "ellipse" to recover the previous trajectory.
print_shape = "square"
square_side = 0.18  # m, centred on the bed
# Ellipse semi-axes (only used when print_shape == "ellipse").
a           = 0.1
b           = 0.07

# Control flows
Color_change            = 0 
Filament_replacement    = 0

#derived system formulas
J_eff           = J_m+m*r_m**2/2 # effective inertia
d_s             = r_m**2/4*(mu_lx + mu_ly + 12*mu_r) #self friction
d_c             = r_m**2/4*(mu_lx - mu_ly + 4*mu_r) #coupled friction
x_c             = dx/2 #center of printbed x-direction
y_c             = dy/2 #center of printbed y-direction
wipe_duration   = 50.0  # Assumed duration of excess-filament wiping [s]
simulation_end  = 1500.0
path_period     = 500.0  # Independent of the total simulation horizon.
new_print_time  = 1.0    # One initial command; no second job is issued.
Omega           = 2 * np.pi / path_period
time_steps      = np.linspace(0, simulation_end, int(np.ceil(simulation_end / 0.5)) + 1)
dt              = time_steps[1] - time_steps[0]

#tolerances 
position_tolerance  =   1e-6
velocity_tolerance  =   1e-6

# State matrix
A_x = np.array([[0, 1 ,0],
              [0, -(d_s+d_c)/J_eff, r_m*K_m/(2*J_eff)],
              [0, -2*K_m/(r_m*L_m), -R_m/L_m]])

# State matrix
A_y = np.array([[0, 1 ,0],
              [0, -(d_s-d_c)/J_eff, r_m*K_m/(2*J_eff)],
              [0, -2*K_m/(r_m*L_m), -R_m/L_m]])

# Input matrix
B = np.array([[0],
              [0],
              [1/L_m]])

# Output matrix
C = np.array([1,0,0])

# Direct feedthrough matrix
D = np.array([0])

t = sp.Symbol("t", real=True)

if print_shape == "square":
    # Four straight sides. Smooth acceleration/deceleration makes velocity
    # and acceleration zero at every corner, so the existing motor model
    # can follow the square without instantaneous changes of direction.
    side_time = path_period / 4
    x_min, y_min = x_c-square_side/2, y_c-square_side/2
    x_max, y_max = x_c+square_side/2, y_c+square_side/2

    def smooth_progress(local_time):
        tau = local_time / side_time
        return 10*tau**3 - 15*tau**4 + 6*tau**5

    x_ref = sp.Piecewise(
        (x_min + square_side*smooth_progress(t), t < side_time),
        (x_max, t < 2*side_time),
        (x_max - square_side*smooth_progress(t-2*side_time), t < 3*side_time),
        (x_min, True)
    )
    y_ref = sp.Piecewise(
        (y_min, t < side_time),
        (y_min + square_side*smooth_progress(t-side_time), t < 2*side_time),
        (y_max, t < 3*side_time),
        (y_max - square_side*smooth_progress(t-3*side_time), t < path_period),
        (y_min, True)
    )
elif print_shape == "ellipse":
    x_ref = x_c + a * sp.cos(Omega*t)
    y_ref = y_c + b * sp.sin(Omega*t)
else:
    raise ValueError("print_shape must be 'square' or 'ellipse'")
dot_x       =  sp.diff(x_ref)
dot_y       =  sp.diff(y_ref)
dotdot_x    =  sp.diff(dot_x)
dotdot_y    =  sp.diff(dot_y)

i_x_ref = 2/(r_m*K_m)*(J_eff*dotdot_x+(d_s+d_c)*dot_x)
i_y_ref = 2/(r_m*K_m)*(J_eff*dotdot_y+(d_s-d_c)*dot_y)

dot_i_x = sp.diff(i_x_ref)
dot_i_y = sp.diff(i_y_ref)

u_x_ff  = L_m*dot_i_x + R_m*i_x_ref + (2*K_m/r_m)*dot_x
u_y_ff  = L_m*dot_i_y + R_m*i_y_ref + (2*K_m/r_m)*dot_y

ux_feedforward = sp.lambdify(t, u_x_ff, "numpy")
uy_feedforward = sp.lambdify(t, u_y_ff, "numpy")

x0_values = np.array([x_ref.subs(t,0), dot_x.subs(t,0), i_x_ref.subs(t,0)], dtype=float)
y0_values = np.array([y_ref.subs(t,0), dot_y.subs(t,0), i_y_ref.subs(t,0)], dtype=float)

nozzle0 = np.array([
    T_ref,
    u_n_eq / R_n,
    u_n_eq
])

z0 = np.concatenate((x0_values, y0_values, nozzle0))

i = 0

def dynamics(t_now, z, ux, uy, un):
    state_x = z[0:3]
    state_y = z[3:6]
    T_nozzle, i_L, v_n = z[6:9]

    derivative_x = A_x @ state_x + B[:, 0] * ux
    x_output = C @ state_x

    derivative_y = A_y @ state_y + B[:, 0] * uy
    y_output = C @ state_y

    dot_T = (v_n**2/R_n-(T_nozzle-T_e)/R_tn)/C_tn

    dot_i_L = (un - v_n) / L_n
    dot_v_n = (i_L - v_n / R_n) / C_n

    dz = np.concatenate((
        derivative_x,
        derivative_y,
        [dot_T, dot_i_L, dot_v_n]
    ))
    return dz

def controller(mode, z, path_time, horizontal_confirmed, vertical_confirmed, return_x_reached):
    if mode == Mode.PRINTING:
        ux = float(ux_feedforward(path_time))
        uy = float(uy_feedforward(path_time))
    elif mode in (Mode.MOVING_TO_WIPE, Mode.PARKING):
        if horizontal_confirmed:
            ux = 0.0
            uy = -0.1
        elif vertical_confirmed:
            ux = 0.0
            uy = 0.0
        else:
            ux = -0.1
            uy = 0.0
    elif mode == Mode.WIPING:
        ux = 0.0
        uy = 0.0
    elif mode == Mode.RETURNING:
        if not return_x_reached:
            ux = 0.1
            uy = 0.0
        else:
            ux = 0.0
            uy = 0.1
    elif mode == Mode.FINISHED:
        ux = 0.0
        uy = 0.0
    else:
        ux = 0.0
        uy = 0.0

    return ux, uy

def main():
    initial_state = np.zeros_like(z0)
    initial_state[6] = T_e  # Parked at (0,0), cold nozzle, no electrical energy.
    z_history = [initial_state]
    un = 0.0
    mode = Mode.WAITING

    path_time = 0.0
    mode_time = 0.0
    saved_position = None
    horizontal_confirmed = False
    vertical_confirmed = False
    replacement_triggered = False
    return_x_reached = False
    new_print_triggered = False
    completed_jobs = 0
    job_id = -1  # First accepted print command creates job 0.
    job_history = []

    filament_color = "tab:red"
    pending_color_change = False

    segment_modes = []
    segment_colors = []
    mode_history = [mode]
    event_log = []

    for n in range(len(time_steps) - 1):
        z = z_history[n].copy()
        t_now = time_steps[n]
        dt = time_steps[n + 1] - time_steps[n]

        switch_off_time = path_period / 3  # 2,5 seconden
        Color_change = (
            mode == Mode.PRINTING
            and path_time >= switch_off_time
            and not replacement_triggered
        )

        if Color_change:
            event_log.append((t_now, "Colour change requested"))

        # 1. Controleer guards.
        if (mode == Mode.PRINTING) and (Filament_replacement or Color_change):
            mode = Mode.MOVING_TO_WIPE
            pending_color_change = bool(Color_change)
            saved_position = z[[0, 3]].copy() #save x and y
            saved_motion_state = z[:6].copy()
            mode_time = 0.0
            replacement_triggered = True

        if mode in (Mode.MOVING_TO_WIPE, Mode.PARKING):
            if(z[0] <= 0):
                z[0] = 0.0
                z[1] = 0.0
            if (z[3] <= 0):
                z[3] = 0.0
                z[4] = 0.0
            
            if np.hypot(z[0], z[3]) < position_tolerance and np.hypot(z[1], z[4]) < velocity_tolerance:
                if mode == Mode.PARKING:
                    mode = Mode.WAITING
                    event_log.append((t_now, "Parked: waiting for a new print command"))
                else:
                    mode = Mode.WIPING
                    event_log.append((t_now, "Parked: wiping started"))
                mode_time = 0.0
            
            if(z[0] <= position_tolerance and abs(z[1]) < velocity_tolerance):
                horizontal_confirmed = True
            
            if(z[3] <= position_tolerance and abs(z[4]) < velocity_tolerance):
                vertical_confirmed = True

        if mode == Mode.WIPING:
            un = u_n_eq  # Keep the nozzle hot while wiping.
            if mode_time >= wipe_duration:
                event_log.append((t_now, "Wiping completed / heater off"))
                mode_time = 0.0
                mode = Mode.LOADING

        if (mode == Mode.LOADING):
            un = 0.0
            if pending_color_change:
                filament_color = "tab:purple"
                pending_color_change = False
            if z[6] <= 333: 
                event_log.append((t_now, "Heater switched on"))
                mode = Mode.HEATING
                mode_time = 0.0

        if (mode == Mode.HEATING):
            un = u_n_eq

            if z[6] >= T_ref - 1.0:
                event_log.append((t_now, "Nozzle hot"))
                mode = Mode.RETURNING
                mode_time = 0.0
                return_x_reached = False

        if mode == Mode.PRINTING and path_time >= path_period:
            completed_jobs += 1
            event_log.append((t_now, "Print complete: parking at wipe tower"))
            mode = Mode.PARKING
            mode_time = 0.0
            horizontal_confirmed = False
            vertical_confirmed = False

        if mode == Mode.WAITING:
            # No heater input while awaiting a new command; residual stored
            # electrical energy and nozzle temperature evolve continuously.
            un = 0.0
            if (new_print_time is not None and t_now >= new_print_time
                    and not new_print_triggered):
                event_log.append((t_now, "New print command received"))
                new_print_triggered = True
                job_id += 1
                path_time = 0.0
                saved_motion_state = z0[:6].copy()
                saved_position = z0[[0, 3]].copy()
                return_x_reached = False
                # Reuse HEATING -> RETURNING -> PRINTING for a new job.
                un = u_n_eq
                mode = Mode.HEATING
                mode_time = 0.0

        
        #    Wijzig eventueel mode en saved_position.
        #    Reset mode_time bij een overgang.

        # 2. Bereken inputs voor deze mode.
        ux, uy = controller(
            mode, z, path_time, horizontal_confirmed, vertical_confirmed, return_x_reached
        )

        dot_z = dynamics(t_now, z, ux, uy, un)

        # 3. Integreer dynamics van t_now naar t_now + dt.
        def rhs(t_eval, z_eval):
            # Laat de referentie ook binnen de solverstap doorlopen.
            local_path_time = path_time + (t_eval - t_now)

            ux, uy = controller(
                mode, z_eval, local_path_time, horizontal_confirmed, vertical_confirmed, return_x_reached
            )

            return dynamics(t_eval, z_eval, ux, uy, un)

        integrated_mode = mode

        job_history.append(job_id)
        segment_modes.append(integrated_mode)
        segment_colors.append(filament_color)
        
        solution = solve_ivp(
            rhs,
            (t_now, time_steps[n + 1]),
            z,
            method="Radau",
            rtol=1e-7,
            atol=1e-9
        )

        z = solution.y[:, -1]

        if integrated_mode in (Mode.MOVING_TO_WIPE, Mode.PARKING):
            if z[0] <= 0.0:
                z[0] = 0.0
                z[1] = 0.0

            if z[3] <= 0.0:
                z[3] = 0.0
                z[4] = 0.0

        if mode == Mode.RETURNING:
            target_x, target_y = saved_position

            if not return_x_reached:
                if z[0] >= target_x:
                    z[0] = target_x
                    z[1] = 0.0
                    return_x_reached = True

            else:
                # Houd x voorlopig met een ideale stop op zijn doel.
                z[0] = target_x
                z[1] = 0.0

                if z[3] >= target_y:
                    z[:6] = saved_motion_state.copy()

                    event_log.append((
                        time_steps[n + 1],
                        "Printing resumed"
                    ))
                
                    mode = Mode.PRINTING
                    mode_time = 0.0

        # 4. Werk de timers bij.
        if integrated_mode == Mode.PRINTING:
            path_time += dt

        mode_time += dt
        
        # 5. Bewaar tijd, z, mode en inputs.
        z_history.append(z.copy())
        mode_history.append(mode)

    z_history = np.asarray(z_history)

    filament = simulate_print_jobs(time_steps, z_history, segment_modes, job_history)

    animation = animate_simulation(
        time_steps, z_history, mode_history,
        segment_modes, segment_colors, event_log, filament
    )

    if "--save-gif" in sys.argv:
        output = Path(__file__).resolve().parent / "output"
        output.mkdir(exist_ok=True)
        animation.save(
            output / "Q3_1_animation.gif",
            writer=PillowWriter(fps=25), dpi=100
        )
        print(f"Animation saved: {output / 'Q3_1_animation.gif'}")

    if "--save-frames" in sys.argv:
        export_latex_frames(animation, time_steps, segment_modes, mode_history)

    plt.show()


def export_latex_frames(animation, times, segment_modes, mode_history, count=120):
    """Small event-aware PNG sequence for the LaTeX animate package.

    Sampling is deliberately nonuniform: retain short travel states as well
    as the long thermal stages. The time displayed in each frame is physical.
    """
    directory = Path(__file__).resolve().parent / "frames_Q3_1"
    directory.mkdir(exist_ok=True)
    modes = list(segment_modes) + [mode_history[-1]]
    cuts = [0] + [j for j in range(1, len(modes)) if modes[j] != modes[j-1]] + [len(modes)]
    selected = {0, len(times)-1}
    for start, stop in zip(cuts[:-1], cuts[1:]):
        selected.add(start)
        selected.add((start+stop-1)//2)
        selected.add(stop-1)
    if len(selected) > count:
        raise ValueError("Increase frame count to retain all mode changes.")
    # Fill the remaining budget from a regularly spaced physical-time grid.
    candidates = np.unique(np.linspace(0, len(times)-1, count*4).astype(int))
    while len(selected) < min(count, len(times)):
        remaining = [j for j in candidates if j not in selected]
        if not remaining:
            remaining = [j for j in range(len(times)) if j not in selected]
        chosen = max(remaining, key=lambda j: min(abs(j-k) for k in selected))
        selected.add(chosen)
    selected = sorted(selected)
    for number, frame in enumerate(selected):
        animation._func(frame)
        animation._fig.savefig(directory / f"frame-{number:03d}.png", dpi=110)
    fps = 3
    command = (
        r"\animategraphics[controls,loop,poster=last,width=\linewidth]"
        f"{{{fps}}}{{frames_Q3_1/frame-}}{{000}}{{{len(selected)-1:03d}}}"
    )
    (directory / "animation.tex").write_text(
        "% Requires \\usepackage{animate} in the preamble.\n" + command + "\n",
        encoding="utf-8"
    )
    np.savetxt(directory / "frame_times.csv",
               np.column_stack((np.arange(len(selected)), times[selected])),
               delimiter=",", header="frame,time_s", comments="")
    animation._func(0)
    print(f"Exported {len(selected)} LaTeX frames to {directory}")
    print(command)


# Same ABS and heat-transfer assumptions as Q1_6.py (SI units).
FILAMENT_K = 0.177                 # W/(m K)
FILAMENT_RHO = 1050.0              # kg/m^3
FILAMENT_CP = 2080.0               # J/(kg K)
FILAMENT_HEIGHT = 2e-4             # m; square cross-section
FILAMENT_AIR = 25.0                # W/(m^2 K)
FILAMENT_BED = 50.0                # W/(m^2 K), assumed contact coefficient
BED_TEMPERATURE = 105.0 + 273.15   # K
FILAMENT_CELL_SIZE = 0.5e-3        # m


def simulate_print_jobs(times, states, segment_modes, job_history):
    """Separate single-layer prints; previous object is removed for a new job.

    Do not connect the last cell of job 1 to the first cell of job 2. Each
    completed print cools until the next job; the heatmap then shows the new
    object only. No inter-layer deposition/contact model is assumed.
    """
    job_history = np.asarray(job_history)
    jobs = []
    for job in np.unique(job_history):
        selected = job_history == job
        if not any(selected[j] and mode == Mode.PRINTING
                   for j, mode in enumerate(segment_modes)):
            continue
        masked_modes = [mode if selected[j] else Mode.HEATING
                        for j, mode in enumerate(segment_modes)]
        result = simulate_filament(times, states, masked_modes)
        next_job = np.flatnonzero(job_history > job)
        if len(next_job):
            result["history"][next_job[0]:] = np.nan
        jobs.append(result)
    return dict(
        positions=np.concatenate([job["positions"] for job in jobs]),
        history=np.concatenate([job["history"] for job in jobs], axis=1),
        closure_time=jobs[-1]["closure_time"],
        job_count=len(jobs)
    )


def simulate_filament(times, states, segment_modes):
    """One-way coupling: deposited material receives nozzle temperature.

    Equal-length cells follow the actual PRINTING polyline. Travel/wiping
    deposits no material on this path. A pause creates no new cells, but
    existing cells keep cooling. Thermal feedback on the nozzle is neglected.
    """
    printing = np.array([mode == Mode.PRINTING for mode in segment_modes])
    distances = np.linalg.norm(np.diff(states[:, [0, 3]], axis=0), axis=1)
    intervals = np.flatnonzero(printing & (distances > 1e-12))
    if not len(intervals):
        return dict(positions=np.empty((0, 2)),
                    history=np.empty((len(times), 0)), closure_time=None)

    lengths = distances[intervals]
    ends = np.cumsum(lengths)
    starts = np.r_[0.0, ends[:-1]]
    total_length = ends[-1]
    count = int(np.ceil(total_length / FILAMENT_CELL_SIZE))
    ds = total_length / count
    edges = np.arange(1, count + 1) * ds
    centres = edges - ds / 2

    # Locate cells along printing intervals only, never along parking travel.
    edge_interval = np.minimum(np.searchsorted(ends, edges), len(intervals)-1)
    centre_interval = np.minimum(np.searchsorted(ends, centres), len(intervals)-1)
    edge_fraction = np.clip(
        (edges - starts[edge_interval]) / lengths[edge_interval], 0, 1
    )
    centre_fraction = np.clip(
        (centres - starts[centre_interval]) / lengths[centre_interval], 0, 1
    )
    edge_indices = intervals[edge_interval]
    centre_indices = intervals[centre_interval]
    deposition = (times[edge_indices]
                  + edge_fraction * (times[edge_indices+1] - times[edge_indices]))
    initial_temperatures = (states[edge_indices, 6]
                           + edge_fraction * (states[edge_indices+1, 6]
                                              - states[edge_indices, 6]))
    positions = (states[centre_indices][:, [0, 3]]
                 + centre_fraction[:, None]
                 * (states[centre_indices+1][:, [0, 3]]
                    - states[centre_indices][:, [0, 3]]))

    finished = any(mode in (Mode.PARKING, Mode.WAITING) for mode in segment_modes)
    endpoint_gap = np.linalg.norm(
        states[intervals[-1]+1, [0, 3]] - states[intervals[0], [0, 3]]
    )
    # Closing a cell-sized gap is a discretisation approximation. No periodic
    # connection is made during the interruption or for an unfinished path.
    closed = finished and endpoint_gap <= 2 * ds and count > 2
    closure_time = deposition[-1] if closed else None

    diffusivity = FILAMENT_K / (FILAMENT_RHO * FILAMENT_CP)
    beta_air = 3 * FILAMENT_AIR / (FILAMENT_RHO * FILAMENT_CP * FILAMENT_HEIGHT)
    beta_bed = FILAMENT_BED / (FILAMENT_RHO * FILAMENT_CP * FILAMENT_HEIGHT)
    safe_dt = 0.8 / (2 * diffusivity / ds**2 + beta_air + beta_bed)

    # Include activation and output times exactly; explicit Euler steps satisfy
    # the diffusion-plus-loss positivity bound, including both periodic ends.
    thermal_times = np.unique(np.r_[
        times, deposition,
        np.linspace(times[0], times[-1],
                    int(np.ceil((times[-1]-times[0]) / safe_dt)) + 1)
    ])
    output_indices = np.searchsorted(thermal_times, times)
    history = np.full((len(times), count), np.nan)
    temperature = np.empty(0)
    next_cell = 0
    output_row = 1
    for j in range(len(thermal_times)-1):
        start, end = thermal_times[j:j+2]
        step = end - start
        if len(temperature):
            conduction = np.zeros_like(temperature)
            if closed and start >= closure_time:
                conduction = diffusivity * (
                    np.roll(temperature, 1) - 2*temperature
                    + np.roll(temperature, -1)
                ) / ds**2
            else:
                exchange = diffusivity * np.diff(temperature) / ds**2
                conduction[:-1] += exchange
                conduction[1:] -= exchange
            temperature += step * (
                conduction - beta_air*(temperature-T_e)
                + beta_bed*(BED_TEMPERATURE-temperature)
            )
        stop = np.searchsorted(deposition, end, side="right")
        if stop > next_cell:
            temperature = np.r_[temperature, initial_temperatures[next_cell:stop]]
            next_cell = stop
        if output_row < len(times) and j+1 == output_indices[output_row]:
            history[output_row, :len(temperature)] = temperature
            output_row += 1

    equilibrium = (beta_air*T_e + beta_bed*BED_TEMPERATURE) / (beta_air+beta_bed)
    print(f"Filament: {count} cells; ds = {ds*1000:.3f} mm; "
          f"maximum thermal step = {np.diff(thermal_times).max():.4f} s")
    print(f"Filament equilibrium: {equilibrium-273.15:.2f} deg C; "
          f"ring closed: {closed}")
    return dict(positions=positions, history=history,
                deposition_times=deposition, initial_temperatures=initial_temperatures,
                closure_time=closure_time, equilibrium=equilibrium, delta_s=ds)


def animate_simulation(times, states, mode_history,
                       segment_modes, segment_colors, event_log, filament):
    """Animate stored results; the physical simulation is not recalculated."""
    fig = plt.figure(figsize=(12, 10))
    grid = fig.add_gridspec(
        2, 2, height_ratios=[1.1, 1],
        left=0.09, right=0.93, bottom=0.08, top=0.88,
        wspace=0.32, hspace=0.65
    )
    ax1 = fig.add_subplot(grid[0, 0])
    ax_heat = fig.add_subplot(grid[0, 1])
    ax2 = fig.add_subplot(grid[1, 0])

    status = fig.text(
        0.08, 0.985, "", va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray")
    )
    # Replacement cycle plus completion -> parking -> standby -> new job.
    ax_fsm = fig.add_subplot(grid[1, 1])
    ax_fsm.set(xlim=(-1.0, 9.5), ylim=(-4.0, 3.7), title="Operating state")
    ax_fsm.set_aspect("equal")
    ax_fsm.axis("off")
    locations = {
        Mode.PRINTING: (1.0, 2.5),
        Mode.MOVING_TO_WIPE: (4.5, 2.5),
        Mode.WIPING: (8.0, 2.5),
        Mode.LOADING: (8.0, 0.0),
        Mode.HEATING: (4.5, 0.0),
        Mode.RETURNING: (1.0, 0.0),
        Mode.PARKING: (1.0, -2.5),
        Mode.WAITING: (4.5, -2.5),
    }
    names = ["Printing", "Moving to wipe", "Wiping", "Loading",
             "Heating", "Returning", "Parking", "Waiting for new print"]
    state_number = {
        Mode.WAITING: 0, Mode.HEATING: 1, Mode.RETURNING: 2,
        Mode.PRINTING: 3, Mode.MOVING_TO_WIPE: 4, Mode.WIPING: 5,
        Mode.LOADING: 6, Mode.PARKING: 7,
    }
    state_nodes, state_labels = {}, {}
    for index, (state, position) in enumerate(locations.items()):
        circle = Circle(position, 0.30, facecolor="0.94",
                        edgecolor="0.45", linewidth=1.2, zorder=3)
        ax_fsm.add_patch(circle)
        state_nodes[state] = circle
        state_labels[state] = ax_fsm.text(
            *position, f"$q_{state_number[state]}$", ha="center", va="center", zorder=4
        )
        offset = -0.60 if state == Mode.PRINTING else 0
        label_y = position[1]+0.48 if state == Mode.HEATING else position[1]-0.43
        if state == Mode.WIPING:
            ax_fsm.text(position[0]+0.42, position[1], names[index],
                        ha="left", va="center", fontsize=8)
        else:
            ax_fsm.text(position[0]+offset, label_y, names[index],
                        ha="center", va="bottom" if state == Mode.HEATING else "top", fontsize=8)

    transitions = [
        (Mode.PRINTING, Mode.MOVING_TO_WIPE, r"$e_{\mathrm{replace}}$"),
        (Mode.MOVING_TO_WIPE, Mode.WIPING, r"$e_{\mathrm{park}}$"),
        (Mode.WIPING, Mode.LOADING, r"$e_{\mathrm{wipe}}$"),
        (Mode.LOADING, Mode.HEATING, r"$e_{\mathrm{load}}$"),
        (Mode.HEATING, Mode.RETURNING, r"$e_{\mathrm{hot}}$"),
        (Mode.RETURNING, Mode.PRINTING, r"$e_{\mathrm{return}}$"),
        (Mode.PARKING, Mode.WAITING, r"$e_{\mathrm{park}}$"),
        (Mode.WAITING, Mode.HEATING, r"$e_{\mathrm{new,print}}$"),
    ]
    for source, target, label in transitions:
        a, b = np.array(locations[source]), np.array(locations[target])
        direction = (b-a)/np.linalg.norm(b-a)
        ax_fsm.add_patch(FancyArrowPatch(
            a+0.31*direction, b-0.31*direction,
            arrowstyle="->", mutation_scale=12, color="0.45"
        ))
        middle = (a+b)/2
        ax_fsm.text(middle[0]+(0.12 if a[1] != b[1] else 0),
                    middle[1]+(0 if a[1] != b[1] else 0.12), label,
                    ha="left" if a[1] != b[1] else "center", fontsize=9)

    # Route completion outside the left edge, away from return -> printing.
    ax_fsm.plot([0.69, -0.70, -0.70, 0.60], [2.5, 2.5, -2.5, -2.5],
                 color="0.55", linewidth=0.9, zorder=0)
    ax_fsm.add_patch(FancyArrowPatch((0.60, -2.5), (0.69, -2.5),
                       arrowstyle="->", mutation_scale=12, color="0.55"))
    ax_fsm.text(-0.82, -0.9, r"$e_{\mathrm{print,complete}}$", rotation=90,
                ha="center", fontsize=8, color="0.4")
    ax_fsm.add_patch(FancyArrowPatch((6.3, -2.5), (4.82, -2.5),
                       arrowstyle="->", mutation_scale=12, color="0.45"))
    ax_fsm.text(8.0, -2.5, "Active state", ha="center", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="#ffe08a", edgecolor="#c28200"))

    ax1.set(xlim=(-0.01, 0.30), ylim=(-0.01, 0.30),
            xlabel="x position [m]", ylabel="y position [m]",
            title="Print head trajectory")
    ax1.set_aspect("equal", adjustable="box")
    ax1.grid(True)

    # One artist per continuous trajectory section keeps dash patterns intact.
    sections = []
    used_labels = set()
    j = 0
    while j < len(segment_modes):
        mode = segment_modes[j]
        color = segment_colors[j]
        end = j + 1
        while (end < len(segment_modes)
               and segment_modes[end] == mode
               and segment_colors[end] == color):
            end += 1

        if mode == Mode.PRINTING:
            style = "-"
            label = ("Printing: new colour" if color == "tab:purple"
                     else "Printing: original colour")
        elif mode in (Mode.MOVING_TO_WIPE, Mode.PARKING):
            color, style, label = "gray", "--", "Moving to wipe tower"
        elif mode == Mode.RETURNING:
            color, style, label = "gray", ":", "Returning to print position"
        else:
            j = end
            continue

        line, = ax1.plot([], [], color=color, linestyle=style,
                         label=label if label not in used_labels else "_nolegend_")
        used_labels.add(label)
        sections.append((j, end, line))
        j = end

    head, = ax1.plot([], [], "o", color="black", markersize=6,
                     label="Print head", zorder=5)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20),
                fontsize=8, ncol=2, borderaxespad=0.0)

    # A separate heatmap preserves the red/purple filament-colour view.
    ax_heat.set(xlim=(-0.01, 0.30), ylim=(-0.01, 0.30),
                xlabel="x position [m]", ylabel="y position [m]",
                title="Deposited filament temperature")
    ax_heat.set_aspect("equal", adjustable="box")
    ax_heat.grid(True)
    heat = ax_heat.scatter([], [], c=[], s=12, cmap="inferno",
                           norm=Normalize(T_e-273.15, T_ref-273.15), zorder=3)
    heat_head, = ax_heat.plot([], [], "o", color="deepskyblue",
                              markeredgecolor="black", markersize=6, zorder=5)
    colorbar_ax = ax_heat.inset_axes([1.04, 0.0, 0.045, 1.0])
    fig.colorbar(heat, cax=colorbar_ax,
                 label="Filament temperature [°C]")

    temperatures = states[:, 6] - 273.15
    ax2.set(xlim=(times[0], times[-1]),
            ylim=(min(T_e - 273.15, temperatures.min()) - 5,
                  max(T_ref - 273.15, temperatures.max()) + 10),
            xlabel="Time [s]", ylabel="Nozzle temperature [°C]",
            title="Nozzle temperature")
    ax2.grid(True)
    temperature_line, = ax2.plot([], [], label="Nozzle temperature")
    temperature_point, = ax2.plot([], [], "o", color="tab:blue")
    cursor = ax2.axvline(times[0], color="black", linestyle="--", alpha=0.5)
    ax2.axhline(T_e - 273.15, color="gray", linestyle=":",
                label="Ambient temperature")
    ax2.legend(loc="center", fontsize=8)

    # Keep every movement sample and every transition, even when skipping
    # samples during long heating/printing periods. Playback is thus adaptive.
    frames = set(range(0, len(times), 4))
    frames.update((0, len(times) - 1))
    for j, mode in enumerate(segment_modes):
        if mode in (Mode.MOVING_TO_WIPE, Mode.PARKING, Mode.RETURNING):
            frames.update((j, j + 1))
        if j == 0 or mode != segment_modes[j - 1]:
            frames.update((max(0, j - 1), j, j + 1))
    for event_time, _ in event_log:
        frames.add(min(int(np.searchsorted(times, event_time)), len(times) - 1))
    frames = sorted(frames)

    # A segment's mode applies from its start until the following sample.
    display_modes = list(segment_modes) + [mode_history[-1]]

    def update(frame):
        for start, end, line in sections:
            stop = min(frame, end)
            if frame < start:
                line.set_data([], [])
            else:
                line.set_data(states[start:stop+1, 0], states[start:stop+1, 3])

        active = np.isfinite(filament["history"][frame])
        heat.set_offsets(filament["positions"][active])
        heat.set_array(filament["history"][frame, active] - 273.15)
        heat_head.set_data([states[frame, 0]], [states[frame, 3]])
        head.set_data([states[frame, 0]], [states[frame, 3]])
        temperature_line.set_data(times[:frame+1], temperatures[:frame+1])
        temperature_point.set_data([times[frame]], [temperatures[frame]])
        cursor.set_xdata([times[frame], times[frame]])
        latest_event = next(
            (name for when, name in reversed(event_log) if when <= times[frame]),
            "Simulation started"
        )
        current_mode = display_modes[frame]
        for state, circle in state_nodes.items():
            selected = state == current_mode
            circle.set_facecolor("#ffe08a" if selected else "0.94")
            circle.set_edgecolor("#c28200" if selected else "0.45")
            circle.set_linewidth(2.8 if selected else 1.2)
            state_labels[state].set_fontweight("bold" if selected else "normal")
        status.set_text(
            f"Time: {times[frame]:.1f} s   |   "
            f"Nozzle: {temperatures[frame]:.1f} °C\n"
            f"Last event: {latest_event}   |   Deposited filament cells: {active.sum()}"
        )
        return (head, heat, heat_head, temperature_line, temperature_point, cursor, status,
                *(line for _, _, line in sections), *state_nodes.values())

    animation = FuncAnimation(
        fig, update, frames=frames, interval=40, blit=False, repeat=False
    )
    # Keep the animation alive for interactive backends.
    fig.animation = animation
    update(0)
    return animation


if __name__ == "__main__":
    main()
