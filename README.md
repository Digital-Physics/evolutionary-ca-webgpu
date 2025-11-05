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
#

To do:  

Add Memory for generalization. 

key-value cache: {12x12_pattern_average: action_sequence_list}. In evolutionary processes 2+, within the same user session, we can now compare a new, unseen target pattern, with the cache patterns previously seen. If we've already seen the pattern, return the key value, the action sequence solution list. If not, do (Approximate) Nearest Neighbor to find top k closest closest pattern matches for games around the length of the new_game_step_count. (Note: we can do a replay of the action sequence stored to get the  states leading up to the final step, if we think that is helpful or necessary, but we won't do that for the moment with this approach.)  

These action sequences that lead to a close pattern match with the new target pattern (as well as other random or constructed action sequences), can act as the starting sequences for further evolutionary exploration. 

Fuzzy memory is okay; it doesn't need to be perfect; it's just initializing the evolutionary process to explore some more. => Should we use a composite pattern that averages the Cellular Automata state over a few time steps? 🤔

(e.g. (1/3) * state_t-2 + (1/3) * state_t-1 + (1/3) * state_t) ? 🤔

Look at time-space trade-offs.

Memory approaches (e.g. save every state at every step vs. save every 5th state vs. save end state only)

recall speed (i.e. hash key, nearest neighbor, approximate nearest neighbor retrieval), 

the recall evaluation metric (TP/(TP + FN) where TP = mutations on surfaced sequences that lead to the solution, and FN = sampled sequences already seen that were not retrieved with ANN or NN, but which lead to the solution. Newly generated sequences that lead to a solution (again, after some mutations to the action sequence over generations) would not be either a TP or a FN (or FP, TN (because it lead to a successful solution)) To do this, we'd need to tag each sequence at the start (e.g. new, retrieved, resampled_but_not_retrieved) and carry it though each generation.),  

the number of additional evolution generations until an exact solution is reached,   

and the overall helpfulness of "memory" (of past action sequences and patterns) to assist in finding solutions to new patterns more quickly.  

New bar chart needed in UI: "Total Unique Action Sequences Seen Across All Evolutions" with the x axis being "Action sequence end length"