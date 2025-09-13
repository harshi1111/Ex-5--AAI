<H3>NAME - HARSHITHA V</H3>
<H3>REGISTER NO - 212223230074</H3>
<H3></H3>
<H1 ALIGN =CENTER>EX. NO.5 Implementation of Kalman Filter</H1>
<H3>Aim:</H3> To Construct a Python Code to implement the Kalman filter to predict the position and velocity of an object.
<H3>Algorithm:</H3>
Step 1: Define the state transition model F, the observation model H, the process noise covariance Q, the measurement noise covariance R, the initial state estimate x0, and the initial error covariance P0.<BR>
Step 2:  Create a KalmanFilter object with these parameters.<BR>
Step 3: Simulate the movement of the object for a number of time steps, generating true states and measurements. <BR>
Step 3: For each measurement, predict the next state using kf.predict().<BR>
Step 4: Update the state estimate based on the measurement using kf.update().<BR>
Step 5: Store the estimated state in a list.<BR>
Step 6: Plot the true and estimated positions.<BR>

### Program:
```
import numpy as np
class KalmanFilter:
    def __init__(self, F, H, Q, R, x0, P0):
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P

dt = 0.1
F = np.array([[1, dt], [0, 1]])
H = np.array([[1, 0]])
Q = np.diag([0.1, 0.1])
R = np.array([[1]])
x0 = np.array([0, 0])
P0 = np.diag([1, 1])

true_states = []
measurements = []
true_x = np.array([0, 0])
for t in range(100):
    true_x = F @ true_x + np.random.normal(0, np.sqrt(Q.diagonal()))
    true_states.append(true_x.copy())
    measurements.append(H @ true_x + np.random.normal(0, np.sqrt(R.diagonal())))
    
kf = KalmanFilter(F, H, Q, R, x0, P0)
est_states = []
for z in measurements:
    kf.predict()
    kf.update(z)
    est_states.append(kf.x)

import matplotlib.pyplot as plt
plt.plot([s[0] for s in true_states], label='true')
plt.plot([s[0] for s in est_states], label='estimate')
plt.legend()
plt.show()
```

### Output:
<img width="601" height="421" alt="image" src="https://github.com/user-attachments/assets/f1a3aa9d-39d1-483e-a709-b1359dabc26c" />


<H3>Results:</H3>
Thus, Kalman filter is implemented to predict the next position and   velocity in Python



