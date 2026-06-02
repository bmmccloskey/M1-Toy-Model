import numpy as np
import matplotlib.pyplot as plt

# M1 as a 2D linear rotational dynamical system

# State x = [x1, x2] lives in a (tiny) neuron state space!
# The activity trajectory over time is modeled by a system of linear, autonomous differential equations: dx/dt = Ax. 
# Reach (movement) direction is encoded as a starting point, all with the same A.

omega = 10     # ROTATION FREQUENCY (rad/s) = imaginary part of eigenvalues
lam   = 1      # DAMPING (1/s) = magnitude of real part of eigenvalues (real part is -lam, so lam>0 -> decay)
x0  = [1,0]    # initial state

# NOTE: Euler's method is only stable when |1 + dt*(-lam + i*omega)| < 1 (aka dt < 2*lam / (lam**2 + omega**2). High omega or low lambda 
# shrinks this. Exceeding it will cause the trajectory to spiral outward instead of decay. Lowering dt will probably fix it.

A = np.array([[-lam, -omega],
              [ omega, -lam]])

print("eigenvalues:", np.linalg.eigvals(A)) # -lam ± i*omega

def simulate(x0, T=4, dt=4e-3):         # initial state; total simulation time; time step
    n = int(T / dt)                     # total steps
    X = np.zeros((n, 2))                # pre allocate array (n rows, 2 columns) for the state at each time step
    X[0] = x0     

    for t in range(n - 1):
        X[t+1] = X[t] + dt * (A @ X[t]) # next state = the current state + time step * dx/dt (Euler's method)
    return X

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4.5))
X = simulate(x0)
ax1.plot(X[:, 0], X[:, 1])              # state space trajectory
ax1.plot(*x0, 'o')                      # initial condition point
ax2.plot(X[:, 0])                       # dim 1 over time
ax3.plot(X[:, 1])                       # dim 2 over time

ax1.set_aspect('equal'); ax1.set_title('neuron state space')
ax1.set_xlabel('dim 1 activity'); ax1.set_ylabel('dim 2 activity')
ax2.set_title('dim 1 readout over time')
ax2.set_xlabel('time step index'); ax2.set_ylabel('dim 1 activity')
ax3.set_title('dim 2 readout over time')
ax3.set_xlabel('time step index'); ax3.set_ylabel('dim 2 activity')
plt.tight_layout(); plt.show()