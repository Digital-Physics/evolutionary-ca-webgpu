# Evolving Action Sequence Solutions to a Pattern Matching Game in Conway's Game of Life

This is a Python-to-TypeScript-and-WepGPU port of a cellular automaton game that is learnable with evolutionary algorithms.

There is a WebGPU compute shader that evaluates action sequences.

This parallelization of the evolutionary process will be compared with regular web workers, which will most likely be the architecture chosen for the final https://www.nets-vs-automata.net/rl.html website-integrated version of this project.  

Here's the stand-alone WebGPU website. Make sure to have WebGPU options turned on and your in a compatible browser.

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

