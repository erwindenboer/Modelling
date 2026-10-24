import control as co
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# System parameters
r_m     = 11e-3 # motor radius
K_m     = 0.07  # motor constant
L_m     = 10e-3 # motor inductance
m       = 60e-3 # printer head mass
J_m     = 8e-6  # motor inertia
R_m     = 90    # motor resistance
mu_lx   = 0.002 # linear bearing friction, x-direction
mu_ly   = 0.001 # linear bearing friction, y-direction
mu_r    = 0.003 # Rolling bearing friction
dx      = 0.29  # Print bed length
dy      = 0.29  # Print bed height

#derived system formulas
J_eff   = J_m+m*r_m**2/2 # effective inertia
d_s     = r_m**2/4*(mu_lx + mu_ly + 12*mu_r) #self friction
d_c     = r_m**2/4*(mu_lx - mu_ly + 4*mu_r) #coupled friction
x_c     = dx/2 #center of printbed x-direction
y_c     = dy/2 #center of printbed y-direction
Time    = 10 #Omlooptijd
Omega   = 2*np.pi/Time

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

# Create the state-space model
sys_x = co.ss(A_x, B, C, D)

# Create the state-space model
sys_y = co.ss(A_y, B, C, D)

t = sp.Symbol("t", real=True)

data = np.genfromtxt(
    "Efteling.csv",
    delimiter=",",
    names=True,
    max_rows=200
)

k = data["turns_n"]
c = data["re"] + 1j * data["im"]

Omega = 2 * np.pi / Time
time = np.linspace(0, Time, 2000)

# Iedere rij is één Four term; tel de rijen op
z = np.sum(
    c[:, None] * np.exp(1j * k[:, None] * Omega * time[None, :]),
    axis=0
)

x = z.real
y = z.imag

x_raw = sp.Add(*[
    float(ck.real) * sp.cos(int(ki) * Omega * t)
    - float(ck.imag) * sp.sin(int(ki) * Omega * t)
    for ki, ck in zip(k, c)
])

y_raw = sp.Add(*[
    float(ck.real) * sp.sin(int(ki) * Omega * t)
    + float(ck.imag) * sp.cos(int(ki) * Omega * t)
    for ki, ck in zip(k, c)
])

scale = 0.20 / max(np.ptp(x), np.ptp(y))

x_mid = (x.max() + x.min()) / 2
y_mid = (y.max() + y.min()) / 2

x_ref = 0.145 + scale * (x_raw - x_mid)
y_ref = 0.145 - scale * (y_raw - y_mid)

a = 0.1
b = 0.07

#x_ref       = x_c + a * sp.cos(Omega*t)
#y_ref       = y_c + b * sp.sin(Omega*t)
dot_x       =  sp.diff(x_ref)
dot_y       =  sp.diff(y_ref)
dotdot_x    =  sp.diff(dot_x)
dotdot_y    =  sp.diff(dot_y)

i_x = 2/(r_m*K_m)*(J_eff*dotdot_x+(d_s+d_c)*dot_x)
i_y = 2/(r_m*K_m)*(J_eff*dotdot_y+(d_s-d_c)*dot_y)

dot_i_x = sp.diff(i_x)
dot_i_y = sp.diff(i_y)

u_x = L_m*dot_i_x+R_m*i_x+(2*K_m/r_m)*dot_x
u_y = L_m*dot_i_y+R_m*i_y+(2*K_m/r_m)*dot_y

x_x0 = [x_ref.subs(t,0), dot_x.subs(t,0), i_x.subs(t,0)]
y_y0 = [y_ref.subs(t,0), dot_y.subs(t,0), i_y.subs(t,0)]

timepts = np.linspace(0, Time, 1000)
ux_function = sp.lambdify(t, u_x, modules="numpy")
ux_values = ux_function(timepts)
x0_values = np.array(x_x0, dtype=float)
tout, x_position = co.forced_response(
    sys_x, timepts, ux_values, x0_values
)

uy_function = sp.lambdify(t, u_y, modules="numpy")
uy_values = uy_function(timepts)
y0_values = np.array(y_y0, dtype=float)
tout, y_position = co.forced_response(
    sys_y, timepts, uy_values, y0_values
)

plt.figure()
plt.plot(x_position, y_position, color = 'r')
plt.xlabel('x position')
plt.ylabel('y position')
plt.title('Printed shape')
plt.xlim(0, dx)
plt.ylim(0, dy)
plt.grid(True)
plt.axis("equal")
plt.show()