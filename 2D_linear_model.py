import numpy as np
import matplotlib.pyplot as plt

# M1 as a 2D linear rotational dynamical system
# State x = [x1, x2] lives in a (tiny) neuron state space!
# dx/dt = Ax -> linear & autonomous 

omega = 10     # ROTATION FREQUENCY (rad/s)
lam   = 1                 # DAMPING (1/s)
x0  = [1,1]

A = np.array([[-lam, -omega],
              [ omega, -lam]])

print("eigenvalues:", np.linalg.eigvals(A))   # -lam ± i*omega

def simulate(x0, T=4, dt=4e-3): # initial state; total simulation time; time step
    n = int(T / dt) # total steps
    X = np.zeros((n, 2)); X[0] = x0 # pre allocate array for the state at each time step
    for t in range(n - 1):
        X[t+1] = X[t] + dt * (A @ X[t])         # next state = the current state + dx/dt * time step (Euler's method)
    return X

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4.5))
X = simulate(x0)
ax1.plot(X[:, 0], X[:, 1])              # state-space trajectory
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