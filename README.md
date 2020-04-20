# gym-mupen64plus

## Configuring Smash
CPU difficulty level, map and character selection is in Smash/discrete_env.py and Smash/smash_env.py

## Environment
### Action Space
[-128, 127] X Joystick  
[-128, 127] Y Joystick  
[0, 1] A  
[0, 1] B  
[0, 1] RB  
[0, 1] LB  
[0, 1] Z  
[0, 1] C  

#### Mapped Action Space
0 - 8 Preset Joystick Movements  
9 A  
10 B  
11 RB  
12 LB  
13 Z  
14 C  

## Setup

The easiest, cleanest, most consistent way to get up and running with this project is via [`Docker`](https://docs.docker.com/). These instructions will focus on that approach.

### With Docker

1. Run the following command to build the project's docker image

    > You should substitute the placeholders between `< >` with your own values.

    ```sh
    docker build -t <image_name>:<tag> .
    ```
    ```sh
    # Example:
    docker build -t bz/gym-mupen64plus:0.0.5 .
    ```

## Example Agents

### Simple Test:
A simple example to test if the environment is up-and-running:
```sh
docker run -it \
  --name test-gym-env \
  -p 5900 \
  --mount source="$(MY_ROM_PATH)",target=/src/gym-mupen64plus/gym_mupen64plus/ROMs,type=bind \
  bz/gym-mupen64plus:0.0.5 \ # This should match the image & tag you used during setup
  python verifyEnv.py
```

```python
#!/bin/python
import gym, gym_mupen64plus

env = gym.make('Mario-Kart-Luigi-Raceway-v0')
env.reset()

for i in range(88):
    (obs, rew, end, info) = env.step([0, 0, 0, 0, 0]) # NOOP until green light

for i in range(100):
    (obs, rew, end, info) = env.step([0, 0, 1, 0, 0]) # Drive straight

raw_input("Press <enter> to exit... ")

env.close()
```


## Architecture

### `Mupen64PlusEnv`:

The core `Mupen64PlusEnv` class has been built to handle many of the details of the wrapping and execution of the Mupen64Plus emulator, as well as the implementation of the gym environment. In fact, it inherits from `gym.Env`. The class is abstract and each game environment inherits from it. The game environment subclass provides the ROM path to the base.

#### Initialization:
* starts the controller server using the port specified in the configuration
* starts the emulator process with the provided ROM path (this also uses values from the config file)
* sets up the observation and action spaces (see the [gym documentation](https://gym.openai.com/docs))
    * the observation space is the screen pixels, by default [640, 480, 3]
    * the default action space is the controller mapping provided by `mupen64plus-input-bot`
        * Joystick X-axis (L/R): value from -80 to 80
        * Joystick Y-axis (U/D): value from -80 to 80
        * A Button: value of 0 or 1
        * B Button: value of 0 or 1
        * RB Button: value of 0 or 1

#### Methods:
* `_step(action)` handles taking the supplied action, passing it to the controller server, and reading the new `observation`, `reward`, and `end_episode` values.

* `_observe()` grabs a screenshot of the emulator window and returns the pixel data as a numpy array.

* `_render()` returns the image or opens a viewer depending on the specified mode. Note that calling `_render()` inside a container currently interferes with the emulator display causing the screen to appear frozen, and should be avoided.

* `_close()` shuts down the environment: stops the emulator, and stops the controller server.

* Abstract methods that each game environment must implement:
    * `_navigate_menu()` moves through the game menu from startup to the beginning of an episode.

    * `_get_reward()` determines the reward for each step.

    * `_evaluate_end_state()` determines whether or not the episode is over.

    * `_reset()` resets the environment to begin a new episode.

### `ControllerHTTPServer`:

When initialized, will start an HTTP Server listening on the specified port. The server will listen for `GET` requests, but will wait to respond until `send_controls()` is called. Each time `send_controls()` is called, it will block and wait for the `GET` request to be processed (up to a configured timeout). In other words, the emulator will end up waiting indefinitely for a controller action, essentially waiting for an agent to `step()`.

### `EmulatorMonitor`:

This class simply polls the emulator process to ensure it is still up and running. If not, it prints the emulator process's exit code. Eventually this will also cause the environment to shutdown since the heart of it just died.

