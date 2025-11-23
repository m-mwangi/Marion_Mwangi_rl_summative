# FireRescue RL Agent

This project implements and evaluates multiple reinforcement learning algorithms for controlling a firefighting robot in a custom Gymnasium environment.
The robot must:
- Navigate a continuous 10×10 grid world
- Avoid obstacles
- Manage water resources
- Steer using realistic vehicle constraints
- Locate and extinguish fires
- Complete the mission within a time limit


The project includes implementations of DQN, PPO, A2C, and REINFORCE, trained and compared using consistent evaluation metrics.

## Demo Video

https://youtu.be/JHrab42wkUQ

## Key Features
- Custom Environment: The simulation involves a dynamic environment where the agent must navigate to extinguish fires while avoiding obstacles.
- Action Space: The agent can perform a range of actions, including movement in various directions and accelerating to reach the target more quickly.
- Reward System: The agent is rewarded based on successful fire extinguishing, with penalties for collisions and inefficient movement.

## Hyperparameter Tuning
Various hyperparameters were tuned to achieve better agent performance; learning rate, batch size, buffer size, etc.

## Set Up and Usage
**1. Clone the repository**:
Start by cloning the repository to your local machine. On your terminal, run:
```
git clone https://github.com/m-mwangi/Marion_Mwangi_rl_summative.git
```
**2. Set up a virtual environment**:
```
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```
**3. Install Dependencies**:
```
pip install -r requirements.txt
```
**4. Run Simulation**:
```
python main.py
```
This should display the PyOpenGL visualization of the environment, load the models in the model folder, and see the agent's actions.
