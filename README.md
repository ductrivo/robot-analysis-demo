# Robot Analysis

This project aims to simulate the kinematics and dynamics of robot arms with different controllers. Currently, the project support analysis on *planar open-chain robots* with N revolut joints. We assume that the links' mass are placed at the joints.

<div align="center">
  <img src="https://i.imgur.com/fnhZQth.png" alt="A 3R planar open chain robot." width="50%">
  <p><strong>A 3R planar open chain robot. (Kevin M. Lynch and Frank C. Park, 2017)</strong> </p>
</div>

Note that:

- For kinematics: we compute the joints position $(x_C, y_C)$.
- For dynamics: we use the following dynamical model:
$$\tau = M(\theta)\ddot{\theta} +C(\theta, \dot{\theta}) + G(\theta)$$

## Usage

### Create the robot

```python
from robot_analysis.planar_nr import PlanarRobotNR

# Create robot
links = {
    'l': [1],      # Links lengths [m]
    'm': [1.0],    # Links masses [kg]
    'b': [1],      # Coefficients de frottement visqueux
}
robot = PlanarRobotNR(links)
```

Output:

``` text
Created robot with 1 link(s).
Kinematics: End effector positions are:
    xC=[l1*cos(q1(t))]
    y_C=[l1*sin(q1(t))]
Dynamics: M, C, G matrices are:
    M=Matrix([[1.0*l1**2*m1]])
    C=Matrix([[b1*Derivative(q1(t), t)]])
    G=Matrix([[9.81*l1*m1*cos(q1(t))]])
```

Note that we can access these matrices using `robot.xC_sym`, `robot.yC_sym`, `robot.M_sym`, `robot.C_sym`, `robot.G_sym`, `robot.eqs_sym`.

### Dynamical simulation

We can dynamically simulate the system with a controller as follows.  
For example, with a stepper motor, we can control the robot simply by applying open-loop position control —  
a common approach in small robotic arm projects.  
However, this method may introduce inaccuracies when the motor's torque is insufficient  
(see [`demo1_stepper_motor_case.ipynb`](notebooks/demo1_stepper_motor_case.ipynb)).

``` python
control_method = {
    'method':'openloop',
    't1': 0.5,
    't2':1.5,
    'tf': 2,
    'theta_d': np.array([ np.pi/6]),
    'tau_max': 20.
}
data = robot.simulate(
    x0=np.array([0.0, 0.0]),
    **control_method
)

error = ( 1-data['x_log'][-1,0]/control_method['theta_d'])*100
plot_final(error=error, theta_d = control_method['theta_d'], tau_max=control_method['tau_max'], **data)
```
