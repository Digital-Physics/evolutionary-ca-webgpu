# Evolve a Good Action Sequence for a Pattern Matching Game in Conway's "Game of Life"

This is a Python-to-TypeScript-and-WepGPU port of a cellular automaton game that is learnable with evolutionary algorithms.  

Note: The mutations are done on the action sequences with the highest fitness metrics, not any lower-level "genetic" level (e.g. an algorithmic action sequence generator, an agent with an action policy given a state, etc.) that would be used as a precursor for generating the actual action sequence we're interested in that will solve the pattern matching game.

The code uses a WebGPU compute shader that evaluates action sequences and returns a fitness score. This parallelization approach will be compared with regular web workers, which will most likely be the architecture chosen for the version that eventually gets integrated into https://www.nets-vs-automata.net/rl.html (for broadest accessibility across non-WebGPU browsers).  

Make sure you have WebGPU options turned on and your in a compatible browser like Chrome if you want explore the WebGPU website. 

https://evolutionary-ca-webgpu.onrender.com

#

### build
```
tsc
```

### serve
```
python3 -m http.server
```

