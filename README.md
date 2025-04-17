# Robot Analysis

This project aims to simulate the kinematics and dynamics of robot arms with different controllers. Currently, the project support analysis on *planar open-chain robots* with N revolut joints. We assume that the links' mass are placed at the joints.

<div align="center">
  <img src="https://i.imgur.com/fnhZQth.png" alt="A 3R planar open chain robot." width="50%">
  <p><strong>A 3R planar open chain robot. (Kevin M. Lynch and Frank C. Park, 2017)</strong> </p>
</div>

# Usage

To create the robot:

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