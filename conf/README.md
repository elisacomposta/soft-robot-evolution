# Configuration files description

## test short
Short test in the Walking environment. It generates:
* 4 random individuals
* 6 individuals mutating the ones stored in the map

## walker
Standard experiment in the Walking task. It generates:
* 500 random individuals
* 1500  individuals mutating the ones stored in the map

## pusher trained in carrier
Takes the best designs of a pusher experiment and evaluates it in the carrier task, optimizing the controller in the new task.

## pusher validated in carrier
Takes the best individuals of a pusher experiment and validates it in the carrier task.
It uses both the structure and the controller already optimized.<br>
_Note_: This is possible only when the two task have the same observation space.<br>
To know more about the observation spaces of the experiments see evogym [documentation](https://evolutiongym.github.io/)

