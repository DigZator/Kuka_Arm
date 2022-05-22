# Todo

* increase number of steps in episode - changed to 5k from 2k     
* max_steps in each episode too small     
* HER says that it improves performance on sparse/binary rewards, rewards here are dense however      
* need to inherit from gym.GoalEnv to use HER, that requires a change in how observation is structured


## Installation

Create virtualenv and install dependencies from requirements.txt        
```
pip install -r requirements.txt
```

To install the environment      
```
pip install -e .
```