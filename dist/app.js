"use strict";
// src/app.ts
// transpiled to dist/app.js
// Evolutionary Algorithms for a cellular automaton game
// WebGPU Game of Life simulation & fitness calculations
// Modes: manual, evolve, demo
// CONSTANTS AND CONFIGURATION  
const GRID_SIZE = 12;
const ACTIONS = ['up', 'down', 'left', 'right', 'do_nothing', 'write_0000', 'write_0001', 'write_0010', 'write_0011', 'write_0100', 'write_0101', 'write_0110', 'write_0111', 'write_1000', 'write_1001', 'write_1010', 'write_1011', 'write_1100', 'write_1101', 'write_1110', 'write_1111'];
// Action decoder translates a numeric index to an action object
const ACTION_DECODER = ACTIONS.map((name, index) => {
    const parts = name.split('_');
    if (parts.length === 2 && parts[0] === 'write') {
        const bitString = parts[1];
        // For a pattern "abcd", bits are [a, b, c, d]
        return { type: 'write', patBits: bitString.split('').map(b => parseInt(b, 2)), index };
    }
    return { type: name, patBits: [0, 0, 0, 0], index };
});
const ACTION_LABELS = ['↑', '↓', '←', '→', '∅', ...Array.from({ length: 16 }, (_, i) => (i).toString(16).toUpperCase())];
const ACTION_TYPES = [
    'Move', 'Move', 'Move', 'Move', 'Pass',
    'Write', 'Write', 'Write', 'Write', 'Write', 'Write', 'Write', 'Write',
    'Write', 'Write', 'Write', 'Write', 'Write', 'Write', 'Write', 'Write'
];
// GLOBAL STATE  
let currentMode = 'evolve';
// Manual Mode State
let manualDemoState = new Uint8Array(GRID_SIZE * GRID_SIZE);
let manualTargetState = new Uint8Array(GRID_SIZE * GRID_SIZE);
let manualStep = 0;
let manualAgentX = 5; // Start at (5,5) - upper left of center 2x2
let manualAgentY = 5;
// Shared state between modes
let sharedTargetPattern = new Uint8Array(GRID_SIZE * GRID_SIZE);
// Evolve Mode State
let isRunning = false;
let bestFitness = 0;
let bestSequence = null;
const currentInitialState = new Uint8Array(GRID_SIZE * GRID_SIZE);
let currentGeneration = 0;
let top5Sequences = [];
// Demo Mode State
let demoPlaybackState = null;
let demoPlaybackStep = 0;
let demoInterval = null;
let demoAgentX = 5;
let demoAgentY = 5;
// WebGPU Context
let gpuDevice = null;
let pipeline = null;
// Statistics for charts
let avgFitnessHistory = [];
let allTimeBestFitnessHistory = [];
let diversityHistory = [];
let uniqueSequencesSeen = new Set();
let countAtFirstPerfectMatch = "N/A";
class Island {
    constructor(size, steps, actionsCount) {
        this.size = size;
        this.steps = steps;
        this.sequences = new Uint32Array(size * steps).map(() => Math.floor(Math.random() * actionsCount));
        this.fitness = new Array(size).fill(0);
        this.bestFitness = 0;
        this.bestSequence = null;
    }
    updateFitness(fitnessArray, startIdx) {
        for (let i = 0; i < this.size; i++) {
            this.fitness[i] = fitnessArray[startIdx + i];
            if (this.fitness[i] > this.bestFitness) {
                this.bestFitness = this.fitness[i];
                this.bestSequence = this.sequences.slice(i * this.steps, (i + 1) * this.steps);
            }
        }
    }
    getTopSequences(k) {
        const indices = Array.from({ length: this.size }, (_, i) => i)
            .sort((a, b) => this.fitness[b] - this.fitness[a])
            .slice(0, Math.min(k, this.size));
        return indices.map(idx => ({
            seq: this.sequences.slice(idx * this.steps, (idx + 1) * this.steps),
            fit: this.fitness[idx]
        }));
    }
    evolve(mutationRate, eliteFrac, actionsCount) {
        const eliteCount = Math.floor(this.size * eliteFrac);
        const sortedIndices = Array.from({ length: this.size }, (_, i) => i)
            .sort((a, b) => this.fitness[b] - this.fitness[a]);
        const newSequences = new Uint32Array(this.size * this.steps);
        // Keep elites
        for (let i = 0; i < eliteCount; i++) {
            const idx = sortedIndices[i];
            newSequences.set(this.sequences.slice(idx * this.steps, (idx + 1) * this.steps), i * this.steps);
        }
        // Generate offspring
        for (let i = eliteCount; i < this.size; i++) {
            let parentA_rank = Math.floor(Math.random() * eliteCount);
            let parentB_rank = Math.floor(Math.random() * eliteCount);
            if (eliteCount > 1) {
                while (parentA_rank === parentB_rank) {
                    parentB_rank = Math.floor(Math.random() * eliteCount);
                }
            }
            const parentA_idx = sortedIndices[parentA_rank];
            const parentB_idx = sortedIndices[parentB_rank];
            const crossPoint = Math.floor(Math.random() * this.steps);
            for (let s = 0; s < this.steps; s++) {
                const val = (s < crossPoint)
                    ? this.sequences[parentA_idx * this.steps + s]
                    : this.sequences[parentB_idx * this.steps + s];
                newSequences[i * this.steps + s] =
                    (Math.random() < mutationRate)
                        ? Math.floor(Math.random() * actionsCount)
                        : val;
            }
        }
        this.sequences = newSequences;
    }
}
// COMPUTE SHADER (WGSL)  
const computeShaderWGSL = `
struct Params {
  gridSize: u32,
  batchSize: u32,
  steps: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> inputStates: array<u32>;
@group(0) @binding(2) var<storage, read> inputSequences: array<u32>;
@group(0) @binding(3) var<storage, read_write> outputStates: array<u32>;
@group(0) @binding(4) var<storage, read> targetPattern: array<u32>;
@group(0) @binding(5) var<storage, read_write> fitness: array<u32>;

fn get_idx(x: u32, y: u32) -> u32 {
  return y * params.gridSize + x;
}

fn wrap_add(a: i32, b: i32, m: i32) -> i32 {
  let v = (a + b) % m;
  if (v < 0) { return v + m; }
  return v;
}

fn apply_action_move(state: ptr<function, array<u32, 146>>, action_index: u32) {
  var ax = (*state)[144u];
  var ay = (*state)[145u];
  let grid_size = params.gridSize;

  if (action_index == 0u) { ay = (ay + grid_size - 1u) % grid_size; }
  else if (action_index == 1u) { ay = (ay + 1u) % grid_size; }
  else if (action_index == 2u) { ax = (ax + grid_size - 1u) % grid_size; }
  else if (action_index == 3u) { ax = (ax + 1u) % grid_size; }
  
  (*state)[144u] = ax;
  (*state)[145u] = ay;
}

fn apply_action_write(state: ptr<function, array<u32, 146>>, action_index: u32) {
  let pat = action_index - 5u;
  let ax = (*state)[144u];
  let ay = (*state)[145u];
  let grid_size_i32 = i32(params.gridSize);

  for (var i: i32 = 0; i < 2; i = i + 1) {
    for (var j: i32 = 0; j < 2; j = j + 1) {
      let bit_index = 3u - u32(i * 2 + j);
      let bit = (pat >> bit_index) & 1u;
      
      let wx = u32(wrap_add(i32(ax) + j, 0, grid_size_i32));
      let wy = u32(wrap_add(i32(ay) + i, 0, grid_size_i32));
      (*state)[get_idx(wx, wy)] = bit;
    }
  }
}

fn conway_step(state: ptr<function, array<u32, 146>>) {
  var next_state: array<u32, 146>;
  let grid_size = params.gridSize;
  let grid_size_i32 = i32(grid_size);

  for (var y: u32 = 0; y < grid_size; y = y + 1) {
    for (var x: u32 = 0; x < grid_size; x = x + 1) {
      var alive_neighbors: u32 = 0u;
      for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
          if (dx != 0 || dy != 0) {
            let nx = u32(wrap_add(i32(x) + dx, 0, grid_size_i32));
            let ny = u32(wrap_add(i32(y) + dy, 0, grid_size_i32));
            alive_neighbors = alive_neighbors + (*state)[get_idx(nx, ny)];
          }
        }
      }
      
      let current_cell = (*state)[get_idx(x, y)];
      var next_cell: u32 = 0u;
      if (current_cell == 1u) {
        if (alive_neighbors == 2u || alive_neighbors == 3u) { next_cell = 1u; }
      } else {
        if (alive_neighbors == 3u) { next_cell = 1u; }
      }
      next_state[get_idx(x, y)] = next_cell;
    }
  }

  for (var i: u32 = 0; i < grid_size * grid_size; i = i + 1) {
    (*state)[i] = next_state[i];
  }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx >= params.batchSize) { return; }

  let grid_cell_count = params.gridSize * params.gridSize;
  let state_offset = idx * grid_cell_count;

  var current_state: array<u32, 146>;
  for (var i: u32 = 0; i < grid_cell_count; i = i + 1) {
    current_state[i] = inputStates[state_offset + i];
  }
  current_state[144u] = 5u;
  current_state[145u] = 5u;

  let sequence_offset = idx * params.steps;
  for (var s: u32 = 0; s < params.steps; s = s + 1) {
    let action_index = inputSequences[sequence_offset + s];

    // 1. Evolve the grid by one step of Conway's Game of Life first
    conway_step(&current_state);
    
    // 2. Then apply user action (move, write, or nothing)
    if (action_index <= 3u) {
        apply_action_move(&current_state, action_index);
    } else if (action_index >= 5u) {
        apply_action_write(&current_state, action_index);
    }
  }

  var matches: u32 = 0u;
  for (var i2: u32 = 0; i2 < grid_cell_count; i2 = i2 + 1) {
    if (current_state[i2] == targetPattern[i2]) {
      matches = matches + 1u;
    }
  }

  // Store raw match count, convert to percentage on CPU
  fitness[idx] = matches;
  
  for (var i3: u32 = 0; i3 < grid_cell_count; i3 = i3 + 1) {
    outputStates[state_offset + i3] = current_state[i3];
  }
}
`;
// CPU-SIDE SIMULATION (for Manual/Demo modes)  
function conwayStep(grid) {
    const size = Math.sqrt(grid.length);
    const nextGrid = new Uint8Array(grid.length);
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            let neighbors = 0;
            for (let dy = -1; dy <= 1; dy++) {
                for (let dx = -1; dx <= 1; dx++) {
                    if (dx === 0 && dy === 0)
                        continue;
                    const nx = (x + dx + size) % size;
                    const ny = (y + dy + size) % size;
                    neighbors += grid[ny * size + nx];
                }
            }
            const current = grid[y * size + x];
            if (current === 1) {
                if (neighbors === 2 || neighbors === 3)
                    nextGrid[y * size + x] = 1;
            }
            else {
                if (neighbors === 3)
                    nextGrid[y * size + x] = 1;
            }
        }
    }
    return nextGrid;
}
function applyAction(state, actionIndex, agentPos) {
    const size = Math.sqrt(state.length);
    const newState = new Uint8Array(state);
    const action = ACTION_DECODER[actionIndex];
    switch (action.type) {
        case 'up':
            agentPos.y = (agentPos.y - 1 + size) % size;
            break;
        case 'down':
            agentPos.y = (agentPos.y + 1) % size;
            break;
        case 'left':
            agentPos.x = (agentPos.x - 1 + size) % size;
            break;
        case 'right':
            agentPos.x = (agentPos.x + 1) % size;
            break;
        case 'write':
            for (let i = 0; i < 2; i++) {
                for (let j = 0; j < 2; j++) {
                    const yy = (agentPos.y + i + size) % size;
                    const xx = (agentPos.x + j + size) % size;
                    newState[yy * size + xx] = action.patBits[i * 2 + j];
                }
            }
            break;
    }
    return newState;
}
function evaluateSequenceToGrid(seq) {
    if (!seq)
        return null;
    const size = GRID_SIZE;
    let state = new Uint8Array(currentInitialState);
    let agent = { x: 5, y: 5 };
    for (const actionIndex of seq) {
        state = new Uint8Array(conwayStep(state));
        state = new Uint8Array(applyAction(state, actionIndex, agent));
    }
    return state;
}
// HELPER FOR STATS  
function calculateMatchPercentage(gridA, gridB) {
    if (!gridB)
        return 0;
    const size = GRID_SIZE * GRID_SIZE;
    if (gridA.length !== size || gridB.length !== size)
        return 0;
    let matches = 0;
    for (let i = 0; i < size; i++) {
        if (gridA[i] === gridB[i]) {
            matches++;
        }
    }
    return (matches / size) * 100;
}
// RENDERING  
function renderGrid(canvas, grid, cellColor = '#4299e1', showAgent = false, agentX = 0, agentY = 0) {
    const size = GRID_SIZE;
    const ctx = canvas.getContext('2d');
    if (!ctx)
        return;
    const cell = canvas.width / size;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = '#e2e8f0';
    ctx.fillStyle = cellColor;
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            if (grid && grid[y * size + x] === 1) {
                ctx.fillRect(x * cell, y * cell, cell, cell);
            }
            ctx.strokeRect(x * cell, y * cell, cell, cell);
        }
    }
    if (showAgent) {
        ctx.strokeStyle = '#dd6b20';
        ctx.lineWidth = 3;
        ctx.strokeRect(agentX * cell, agentY * cell, 2 * cell, 2 * cell);
        ctx.lineWidth = 1;
    }
}
function getUIElements() {
    return {
        population: document.getElementById('population'),
        steps: document.getElementById('steps'),
        generations: document.getElementById('generations'),
        elite: document.getElementById('elite'),
        mut: document.getElementById('mut'),
        vizFreq: document.getElementById('vizFreq'),
        startBtn: document.getElementById('startBtn'),
        stopBtn: document.getElementById('stopBtn'),
        statsDiv: document.getElementById('statsDiv'),
        manualDemoCanvas: document.getElementById('manualDemoCanvas'),
        manualTargetCanvas: document.getElementById('manualTargetCanvas'),
        manualStats: document.getElementById('manualStats'),
        manualMaxStepsLabel: document.getElementById('manualMaxStepsLabel'),
        savePatternBtn: document.getElementById('savePatternBtn'),
        demoTargetCanvas: document.getElementById('demoTargetCanvas'),
        demoBestCanvas: document.getElementById('demoBestCanvas'),
        demoActionsCanvas: document.getElementById('demoActionsCanvas'),
        demoPlaybackLabel: document.getElementById('demoPlaybackLabel'),
        demoPlayBtn: document.getElementById('demoPlayBtn'),
        demoPauseBtn: document.getElementById('demoPauseBtn'),
        demoResetBtn: document.getElementById('demoResetBtn'),
        demoStepBtn: document.getElementById('demoStepBtn'),
        log: document.getElementById('log'),
        evolveBestCanvas: document.getElementById('evolveBestCanvas'),
        evolveCurrentCanvas: document.getElementById('evolveCurrentCanvas'),
        targetVizCanvas: document.getElementById('targetVizCanvas'),
        manualResetBtn: document.getElementById('manualResetBtn'),
        fitnessDistCanvas: document.getElementById('fitnessDistCanvas'),
        diversityCanvas: document.getElementById('diversityCanvas'),
        top5List: document.getElementById('top5List'),
        manualActionsPanel: document.getElementById('manualActionsPanel'),
        startIslands: document.getElementById('startIslands'),
        mergeEvery: document.getElementById('mergeEvery'),
        // topKFromIsland: document.getElementById('topKFromIsland') as HTMLInputElement,
        // keepTopTotal: document.getElementById('keepTopTotal') as HTMLInputElement,
    };
}
// MODE HANDLING  
function handleModeChange() {
    document.querySelectorAll('.mode-content').forEach(el => el.style.display = 'none');
    const selectedModeEl = document.getElementById(`${currentMode}-mode`);
    if (selectedModeEl) {
        selectedModeEl.style.display = 'block';
    }
    const ui = getUIElements();
    if (currentMode === 'manual') {
        // renderGrid(ui.manualTargetCanvas, sharedTargetPattern, '#187537ff');
        generateActionButtons(ui.manualActionsPanel, performManualAction);
        updateManualDisplay();
    }
    else if (currentMode === 'evolve') {
        renderGrid(ui.targetVizCanvas, sharedTargetPattern, '#e29d43ff');
    }
    else if (currentMode === 'demo') {
        // initializeDemoMode calls generateActionButtons
        initializeDemoMode();
    }
}
// MANUAL MODE  
function updateManualDisplay() {
    const ui = getUIElements();
    const maxSteps = Number(ui.steps.value);
    renderGrid(ui.manualDemoCanvas, manualDemoState, '#4299e1', true, manualAgentX, manualAgentY);
    renderGrid(ui.manualTargetCanvas, sharedTargetPattern, '#e29d43ff');
    const matchPercent = calculateMatchPercentage(manualDemoState, sharedTargetPattern);
    let statsHTML = `
    Step: <strong>${manualStep}/${maxSteps}</strong><br>
    Temporary Pattern Match: <strong>${matchPercent.toFixed(1)}%</strong>
  `;
    if (manualStep >= maxSteps) {
        statsHTML += `<br>Final Fitness Score: <strong>${matchPercent.toFixed(1)}%</strong>`;
    }
    ui.manualStats.innerHTML = statsHTML;
}
function initManualMode(ui) {
    const updateMaxSteps = () => {
        ui.manualMaxStepsLabel.textContent = ui.steps.value;
        updateManualDisplay();
    };
    updateMaxSteps();
    ui.steps.addEventListener('change', updateMaxSteps);
    const targetClickHandler = (e) => {
        const canvas = e.currentTarget;
        const rect = canvas.getBoundingClientRect();
        const displayWidth = rect.width;
        const cell = displayWidth / GRID_SIZE;
        // const cell = canvas.width / GRID_SIZE;
        const x = Math.floor((e.clientX - rect.left) / cell);
        const y = Math.floor((e.clientY - rect.top) / cell);
        if (x >= 0 && x < GRID_SIZE && y >= 0 && y < GRID_SIZE) {
            const idx = y * GRID_SIZE + x;
            if (!sharedTargetPattern)
                sharedTargetPattern = new Uint8Array(GRID_SIZE * GRID_SIZE);
            sharedTargetPattern[idx] = 1 - sharedTargetPattern[idx];
            renderGrid(ui.manualTargetCanvas, sharedTargetPattern, '#e29d43ff');
            updateManualDisplay();
        }
    };
    ui.manualTargetCanvas.addEventListener('click', targetClickHandler);
    ui.savePatternBtn.addEventListener('click', () => {
        sharedTargetPattern = new Uint8Array(manualDemoState);
        log(ui.log, 'Saved manual canvas to shared target pattern.');
        updateManualDisplay();
    });
    ui.manualResetBtn.addEventListener('click', () => {
        manualDemoState.fill(0);
        manualStep = 0;
        manualAgentX = 5;
        manualAgentY = 5;
        updateManualDisplay();
        log(ui.log, 'Manual mode reset.');
    });
    // generateActionButtons(ui.manualActionsPanel, performManualAction); // Moved to handleModeChange
    updateManualDisplay();
}
function performManualAction(actionIndex) {
    const ui = getUIElements();
    const maxSteps = Number(ui.steps.value);
    if (manualStep >= maxSteps) {
        log(ui.log, 'Max steps reached. Reset to perform new actions.');
        return; // Already at max steps
    }
    if (actionIndex !== -1) {
        const agentPos = { x: manualAgentX, y: manualAgentY };
        manualDemoState = new Uint8Array(conwayStep(manualDemoState));
        manualDemoState = new Uint8Array(applyAction(manualDemoState, actionIndex, agentPos));
        manualAgentX = agentPos.x;
        manualAgentY = agentPos.y;
        manualStep++;
        updateManualDisplay();
    }
}
function handleManualKeyDown(e) {
    if (currentMode !== 'manual')
        return;
    const key = e.key.toLowerCase();
    let actionIndex = -1;
    if (key.startsWith('arrow')) {
        actionIndex = ['arrowup', 'arrowdown', 'arrowleft', 'arrowright'].indexOf(key);
    }
    else if (key === ' ') {
        actionIndex = 4;
    }
    else if (/^[0-9a-f]$/.test(key)) {
        const patternNum = parseInt(key, 16);
        actionIndex = 5 + patternNum;
    }
    if (actionIndex !== -1) {
        e.preventDefault();
        performManualAction(actionIndex); // Call the refactored function
    }
}
// LOGGING  
function log(el, ...args) {
    if (!el)
        return;
    el.textContent += args.join(' ') + '\n';
    el.scrollTop = el.scrollHeight;
}
// WEBGPU SETUP  
async function initWebGPU() {
    if (gpuDevice)
        return gpuDevice;
    if (!navigator.gpu)
        throw new Error('WebGPU not supported on this browser.');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter)
        throw new Error('No compatible GPUAdapter found.');
    gpuDevice = await adapter.requestDevice();
    return gpuDevice;
}
async function setupComputePipeline() {
    if (pipeline)
        return pipeline;
    const device = await initWebGPU();
    const shaderModule = device.createShaderModule({ code: computeShaderWGSL });
    pipeline = device.createComputePipeline({
        layout: 'auto',
        compute: {
            module: shaderModule,
            entryPoint: 'main',
        },
    });
    return pipeline;
}
function formatSequence(seq, maxLen = 12) {
    const labels = ['↑', '↓', '←', '→', '∅', ...Array.from({ length: 16 }, (_, i) => (i).toString(16).toUpperCase())];
    if (seq.length <= maxLen) {
        return Array.from(seq).map(a => labels[a]).join(' ');
    }
    const half = Math.floor(maxLen / 2);
    const start = Array.from(seq.slice(0, half)).map(a => labels[a]).join(' ');
    const end = Array.from(seq.slice(-half)).map(a => labels[a]).join(' ');
    return `${start} … ${end}`;
}
function updateTop5Display(ui) {
    let html = '<div style="font-family: monospace; font-size: 0.75rem; line-height: 1.4;">';
    html += '<strong>All-Gen. Leaderboard</strong><br>';
    html += '═══════════════════════<br>';
    if (top5Sequences.length === 0) {
        html += 'No sequences yet<br>';
    }
    else {
        top5Sequences.forEach((entry, idx) => {
            const seqStr = formatSequence(entry.sequence, 12);
            html += `#${idx + 1}: ${entry.fitness.toFixed(1)}% (Gen. ${entry.generation})<br>`;
            html += `&nbsp;&nbsp;&nbsp;${seqStr}<br>`;
        });
    }
    html += '</div>';
    ui.top5List.innerHTML = html;
}
// EVOLUTION MODE
async function runEvolution(ui) {
    if (!sharedTargetPattern || sharedTargetPattern.every(cell => cell === 0)) {
        log(ui.log, 'ERROR: No target pattern set. Please draw a pattern on the target canvas first.');
        return;
    }
    isRunning = true;
    ui.startBtn.disabled = true;
    ui.stopBtn.disabled = false;
    const device = await initWebGPU();
    await setupComputePipeline();
    const totalPopSize = Number(ui.population.value);
    const steps = Number(ui.steps.value);
    const generations = Number(ui.generations.value);
    const mutationRate = Number(ui.mut.value);
    const eliteFrac = Number(ui.elite.value);
    const vizFreq = Number(ui.vizFreq.value);
    const startIslands = Math.max(1, Number(ui.startIslands.value));
    const mergeEvery = Number(ui.mergeEvery.value);
    // const topKFromIsland = Number(ui.topKFromIsland.value);
    // const keepTopTotal = Number(ui.keepTopTotal.value);
    // Initialize islands
    let islands = [];
    const basePop = Math.floor(totalPopSize / startIslands);
    const remainder = totalPopSize % startIslands;
    for (let i = 0; i < startIslands; i++) {
        const islandSize = basePop + (i < remainder ? 1 : 0);
        islands.push(new Island(islandSize, steps, ACTIONS.length));
    }
    log(ui.log, `Starting evolution with ${startIslands} islands, total population: ${totalPopSize}`);
    const stateSize = GRID_SIZE * GRID_SIZE;
    const stateBufferSize = totalPopSize * stateSize * 4;
    const seqBufferSize = totalPopSize * steps * 4;
    const inputStatesBuffer = device.createBuffer({
        size: stateBufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    const inputSequencesBuffer = device.createBuffer({
        size: seqBufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    const outputStatesBuffer = device.createBuffer({
        size: stateBufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });
    const targetBuffer = device.createBuffer({
        size: stateSize * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    const fitnessBuffer = device.createBuffer({
        size: totalPopSize * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });
    const fitnessReadBuffer = device.createBuffer({
        size: totalPopSize * 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });
    const paramsBuffer = device.createBuffer({
        size: 3 * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    const initialStates = new Uint32Array(totalPopSize * stateSize);
    device.queue.writeBuffer(inputStatesBuffer, 0, initialStates);
    const targetPattern32 = new Uint32Array(sharedTargetPattern);
    device.queue.writeBuffer(targetBuffer, 0, targetPattern32);
    const paramsArray = new Uint32Array([GRID_SIZE, totalPopSize, steps]);
    device.queue.writeBuffer(paramsBuffer, 0, paramsArray);
    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: paramsBuffer } },
            { binding: 1, resource: { buffer: inputStatesBuffer } },
            { binding: 2, resource: { buffer: inputSequencesBuffer } },
            { binding: 3, resource: { buffer: outputStatesBuffer } },
            { binding: 4, resource: { buffer: targetBuffer } },
            { binding: 5, resource: { buffer: fitnessBuffer } },
        ],
    });
    bestFitness = 0;
    bestSequence = null;
    avgFitnessHistory = [];
    allTimeBestFitnessHistory = [];
    diversityHistory = [];
    top5Sequences = [];
    currentGeneration = 0;
    uniqueSequencesSeen = new Set();
    const totalPossibleSequences = Math.pow(ACTIONS.length, steps);
    for (let gen = 0; gen < generations && isRunning; gen++) {
        currentGeneration = gen;
        // Collect all sequences from islands
        const allSequences = new Uint32Array(totalPopSize * steps);
        let offset = 0;
        for (const island of islands) {
            allSequences.set(island.sequences, offset);
            offset += island.sequences.length;
        }
        device.queue.writeBuffer(inputSequencesBuffer, 0, allSequences);
        const commandEncoder = device.createCommandEncoder();
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(totalPopSize / 64));
        pass.end();
        commandEncoder.copyBufferToBuffer(fitnessBuffer, 0, fitnessReadBuffer, 0, totalPopSize * 4);
        device.queue.submit([commandEncoder.finish()]);
        await fitnessReadBuffer.mapAsync(GPUMapMode.READ);
        const fitnessArray = Array.from(new Uint32Array(fitnessReadBuffer.getMappedRange().slice(0)));
        fitnessReadBuffer.unmap();
        // Convert to percentages
        for (let i = 0; i < totalPopSize; i++) {
            fitnessArray[i] = (fitnessArray[i] / stateSize) * 100;
        }
        // Update island fitness
        offset = 0;
        let island_idx = 0;
        for (const island of islands) {
            island.updateFitness(fitnessArray, offset);
            offset += island.size;
            // Update global best
            if (island.bestFitness > bestFitness) {
                if (island.bestFitness === 100 && bestFitness < 100) {
                    countAtFirstPerfectMatch = uniqueSequencesSeen.size.toLocaleString();
                }
                bestFitness = island.bestFitness;
                bestSequence = island.bestSequence ? new Uint32Array(island.bestSequence) : null;
                log(ui.log, `Gen ${gen}: New best fitness ${bestFitness.toFixed(1)}% from island #${island_idx + 1} with ${island.size} members`);
            }
            island_idx++;
        }
        // Update statistics
        let sumFitness = 0;
        const seqSet = new Set();
        offset = 0;
        for (const island of islands) {
            for (let i = 0; i < island.size; i++) {
                sumFitness += island.fitness[i];
                const seqStr = Array.from(island.sequences.slice(i * steps, (i + 1) * steps)).join(',');
                seqSet.add(seqStr);
                uniqueSequencesSeen.add(seqStr);
            }
        }
        const avgFitness = sumFitness / totalPopSize;
        diversityHistory.push((seqSet.size / totalPopSize) * 100);
        avgFitnessHistory.push(avgFitness);
        allTimeBestFitnessHistory.push(bestFitness);
        // Update top 5 sequences
        for (const island of islands) {
            const topFromIsland = island.getTopSequences(5);
            for (const { seq, fit } of topFromIsland) {
                if (fit > 0) {
                    const seqKey = Array.from(seq).join(',');
                    const exists = top5Sequences.find(e => Array.from(e.sequence).join(',') === seqKey);
                    if (!exists) {
                        top5Sequences.push({
                            sequence: new Uint32Array(seq),
                            fitness: fit,
                            generation: gen
                        });
                    }
                }
            }
        }
        top5Sequences.sort((a, b) => b.fitness - a.fitness);
        top5Sequences = top5Sequences.slice(0, 5);
        // Merge logic
        const shouldMerge = ((gen + 1) % mergeEvery === 0) && (islands.length > 1);
        if (shouldMerge) {
            // Collect top sequences from all islands
            const candidates = [];
            for (const island of islands) {
                // added 
                const topKFromIsland = eliteFrac * island.size;
                const topK = island.getTopSequences(topKFromIsland);
                candidates.push(...topK);
            }
            candidates.sort((a, b) => b.fit - a.fit);
            // const kept = candidates.slice(0, Math.min(keepTopTotal, candidates.length));
            // const kept = candidates;
            // log(ui.log, `Gen ${gen + 1}: Merging ${islands.length} islands. Keeping ${kept.length} top sequences.`);
            log(ui.log, `Gen ${gen + 1}: Merging ${islands.length} islands. Keeping ${candidates.length} top sequences.`);
            // Create new islands (half the count)
            const newIslandCount = Math.max(1, Math.ceil(islands.length / 2));
            const newBasePop = Math.floor(totalPopSize / newIslandCount);
            const newRemainder = totalPopSize % newIslandCount;
            const newIslands = [];
            for (let i = 0; i < newIslandCount; i++) {
                const islandSize = newBasePop + (i < newRemainder ? 1 : 0);
                const newIsland = new Island(islandSize, steps, ACTIONS.length);
                // Seed with kept sequences
                // const numToSeed = Math.min(kept.length, Math.max(1, Math.floor(islandSize / 5)));
                const numToSeed = Math.min(candidates.length, Math.max(1, Math.floor(islandSize / 5)));
                // for (let j = 0; j < numToSeed && j < kept.length; j++) {
                for (let j = 0; j < numToSeed && j < candidates.length; j++) {
                    // const pickIdx = Math.floor(Math.random() * kept.length);
                    const pickIdx = Math.floor(Math.random() * candidates.length);
                    // newIsland.sequences.set(kept[pickIdx].seq, j * steps);
                    newIsland.sequences.set(candidates[pickIdx].seq, j * steps);
                }
                newIslands.push(newIsland);
            }
            islands = newIslands;
            log(ui.log, `Now running with ${islands.length} island(s).`);
        }
        else {
            // Normal evolution within each island
            for (const island of islands) {
                island.evolve(mutationRate, eliteFrac, ACTIONS.length);
            }
        }
        // Visualization
        if (gen % vizFreq === 0 || gen === generations - 1) {
            const bestSeqStr = bestSequence ? formatSequence(bestSequence, 20) : 'N/A';
            const explorationPercent = (uniqueSequencesSeen.size / totalPossibleSequences) * 100;
            ui.statsDiv.innerHTML = `
        Gen: <strong>${gen + 1} / ${generations}</strong><br>
        Islands: <strong>${islands.length}</strong> (sizes: ${islands.map(i => i.size).join(', ')})<br>
        Best Fitness: <strong>${bestFitness.toFixed(1)}%</strong><br>
        Avg Fitness: <strong>${avgFitness.toFixed(1)}%</strong><br>
        <hr style="margin: 4px 0; border-top: 1px solid #e2e8f0;">
        Unique Action Sequences Analyzed: <strong>${uniqueSequencesSeen.size.toLocaleString()}</strong><br>
        Total Possible Action Seq. (actions^steps):<br><strong>${totalPossibleSequences.toLocaleString()}</strong><br>
        Percentage Explored: <strong>${explorationPercent.toFixed(12)}%</strong><br>
        Unique Sequences Before Exact Match: <strong>${countAtFirstPerfectMatch}</strong><br>
        <hr style="margin: 4px 0; border-top: 1px solid #e2e8f0;">
        Best Sequence: <small>${bestSeqStr}</small>
      `;
            renderGrid(ui.evolveBestCanvas, evaluateSequenceToGrid(bestSequence), '#38a169');
            renderGrid(ui.targetVizCanvas, sharedTargetPattern, '#e29d43ff');
            // Show random sequence from random island
            const randomIsland = islands[Math.floor(Math.random() * islands.length)];
            const randomIdx = Math.floor(Math.random() * randomIsland.size);
            const randomSeq = randomIsland.sequences.slice(randomIdx * steps, (randomIdx + 1) * steps);
            renderGrid(ui.evolveCurrentCanvas, evaluateSequenceToGrid(randomSeq), '#4299e1');
            renderFitnessChart(ui.fitnessDistCanvas, avgFitnessHistory, allTimeBestFitnessHistory);
            renderDiversityChart(ui.diversityCanvas, diversityHistory);
            updateTop5Display(ui);
            await new Promise(r => setTimeout(r, 1));
        }
        if (bestFitness === 100) {
            log(ui.log, `Perfect match found in generation ${gen}! Continuing to explore other solutions...`);
        }
    }
    isRunning = false;
    ui.startBtn.disabled = false;
    ui.stopBtn.disabled = true;
    log(ui.log, `Evolution complete. Final island count: ${islands.length}`);
}
// CHART RENDERING  
function renderChart(canvas, datasets, yLabel, xLabel
// yMaxFixed?: number
) {
    const ctx = canvas.getContext('2d');
    if (!ctx || datasets.length === 0 || datasets[0].data.length === 0)
        return;
    // Resizing canvas drawing buffer to match display size to prevent stretching  
    const rect = canvas.getBoundingClientRect();
    if (canvas.width !== rect.width || canvas.height !== rect.height) {
        canvas.width = rect.width;
        canvas.height = rect.height;
    }
    const { width, height } = canvas;
    const p = { t: 30, r: 20, b: 40, l: 50 };
    const xRange = width - p.l - p.r;
    const yRange = height - p.t - p.b;
    const numPoints = datasets[0].data.length;
    ctx.clearRect(0, 0, width, height);
    ctx.font = '12px sans-serif';
    // dynamic Y-axis range calculation
    const allValues = datasets.flatMap(d => d.data);
    let yMin = Math.max(Math.min(...allValues) - 1, 0);
    let yMax = 100;
    // draw horizontal lines
    ctx.strokeStyle = '#e2e8f0';
    ctx.fillStyle = '#718096';
    ctx.lineWidth = 1;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    const numTicks = 5;
    for (let i = 0; i <= numTicks; i++) {
        const frac = i / numTicks;
        const y = p.t + yRange * (1 - frac);
        const val = yMin + (yMax - yMin) * frac;
        ctx.beginPath();
        ctx.moveTo(p.l, y);
        ctx.lineTo(p.l + xRange, y);
        ctx.stroke();
        ctx.fillText(val.toFixed(yMax - yMin < 2 ? 2 : 0), p.l - 8, y);
    }
    // draw data lines
    datasets.forEach(({ data, color }) => {
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let i = 0; i < numPoints; i++) {
            const x = p.l + (xRange * i) / Math.max(1, numPoints - 1);
            const norm = (data[i] - yMin) / (yMax - yMin);
            const y = p.t + yRange * (1 - norm);
            if (i === 0)
                ctx.moveTo(x, y);
            else
                ctx.lineTo(x, y);
        }
        ctx.stroke();
    });
    // draw legend (top-right)
    const legendX = width - p.r - 120;
    let legendY = p.t - 10;
    // compute legend box dimensions
    const legendWidth = 110;
    const legendHeight = datasets.length * 15 + 10;
    // draw subtle translucent background + soft border
    ctx.fillStyle = 'rgba(255, 255, 255, 0.8)'; // soft white background
    ctx.fillRect(legendX, legendY, legendWidth, legendHeight);
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.25)'; // faint border
    ctx.lineWidth = 1;
    ctx.strokeRect(legendX + 0.5, legendY + 0.5, legendWidth - 1, legendHeight - 1);
    // draw each legend entry
    let entryY = legendY + 6;
    datasets.forEach(({ color, label }) => {
        ctx.fillStyle = color;
        ctx.fillRect(legendX + 8, entryY, 10, 10);
        ctx.fillStyle = '#2d3748';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        ctx.fillText(label, legendX + 24, entryY - 1);
        entryY += 15;
    });
    // axis labels
    ctx.fillStyle = '#2d3748';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText(xLabel, p.l + xRange / 2, height - 5);
    ctx.save();
    ctx.translate(15, p.t + yRange / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textBaseline = 'top';
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();
}
function renderFitnessChart(canvas, avg, best) {
    renderChart(canvas, [
        { data: best, color: '#38a169', label: 'Best Fitness' },
        { data: avg, color: '#4299e1', label: 'Avg Fitness' }
        // ], 'Fitness %', 'Generation', 100);
    ], 'Fitness %', 'Generation');
}
function renderDiversityChart(canvas, diversity) {
    renderChart(canvas, [
        { data: diversity, color: '#9f7aea', label: 'Diversity %' }
    ], 'Unique Action Sequence %', 'Generation');
}
// DEMO MODE  (should we add this to stats in Evolve mode too?)
function createActionImage(actionIndex, size = 40) {
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');
    if (!ctx)
        return canvas;
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, size, size);
    ctx.strokeStyle = '#cbd5e0';
    ctx.strokeRect(0, 0, size, size);
    ctx.fillStyle = '#2d3748';
    ctx.strokeStyle = '#2d3748';
    ctx.lineWidth = 2;
    const mid = size / 2;
    const arrowSize = size * 0.6;
    if (actionIndex === 0) { // Up
        ctx.beginPath();
        ctx.moveTo(mid, size * 0.2);
        ctx.lineTo(mid - arrowSize / 3, mid);
        ctx.lineTo(mid - arrowSize / 6, mid);
        ctx.lineTo(mid - arrowSize / 6, size * 0.8);
        ctx.lineTo(mid + arrowSize / 6, size * 0.8);
        ctx.lineTo(mid + arrowSize / 6, mid);
        ctx.lineTo(mid + arrowSize / 3, mid);
        ctx.closePath();
        ctx.fill();
    }
    else if (actionIndex === 1) { // Down
        ctx.beginPath();
        ctx.moveTo(mid, size * 0.8);
        ctx.lineTo(mid - arrowSize / 3, mid);
        ctx.lineTo(mid - arrowSize / 6, mid);
        ctx.lineTo(mid - arrowSize / 6, size * 0.2);
        ctx.lineTo(mid + arrowSize / 6, size * 0.2);
        ctx.lineTo(mid + arrowSize / 6, mid);
        ctx.lineTo(mid + arrowSize / 3, mid);
        ctx.closePath();
        ctx.fill();
    }
    else if (actionIndex === 2) { // Left
        ctx.beginPath();
        ctx.moveTo(size * 0.2, mid);
        ctx.lineTo(mid, mid - arrowSize / 3);
        ctx.lineTo(mid, mid - arrowSize / 6);
        ctx.lineTo(size * 0.8, mid - arrowSize / 6);
        ctx.lineTo(size * 0.8, mid + arrowSize / 6);
        ctx.lineTo(mid, mid + arrowSize / 6);
        ctx.lineTo(mid, mid + arrowSize / 3);
        ctx.closePath();
        ctx.fill();
    }
    else if (actionIndex === 3) { // Right
        ctx.beginPath();
        ctx.moveTo(size * 0.8, mid);
        ctx.lineTo(mid, mid - arrowSize / 3);
        ctx.lineTo(mid, mid - arrowSize / 6);
        ctx.lineTo(size * 0.2, mid - arrowSize / 6);
        ctx.lineTo(size * 0.2, mid + arrowSize / 6);
        ctx.lineTo(mid, mid + arrowSize / 6);
        ctx.lineTo(mid, mid + arrowSize / 3);
        ctx.closePath();
        ctx.fill();
    }
    else if (actionIndex === 4) { // Do nothing (empty set)
        const radius = size * 0.3;
        ctx.beginPath();
        ctx.arc(mid, mid, radius, 0, 2 * Math.PI);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(mid - radius * 0.7, mid - radius * 0.7);
        ctx.lineTo(mid + radius * 0.7, mid + radius * 0.7);
        ctx.stroke();
    }
    else if (actionIndex >= 5) { // Write patterns
        const patternNum = actionIndex - 5;
        const bits = [(patternNum >> 3) & 1, (patternNum >> 2) & 1, (patternNum >> 1) & 1, patternNum & 1];
        const cellSize = size * 0.35;
        const startX = mid - cellSize;
        const startY = mid - cellSize;
        ctx.fillStyle = '#2d3748';
        for (let i = 0; i < 2; i++) {
            for (let j = 0; j < 2; j++) {
                if (bits[i * 2 + j]) {
                    ctx.fillRect(startX + j * cellSize, startY + i * cellSize, cellSize, cellSize);
                }
                ctx.strokeRect(startX + j * cellSize, startY + i * cellSize, cellSize, cellSize);
            }
        }
    }
    return canvas;
}
// Generates the 21 action buttons for Manual mode
function generateActionButtons(container, clickHandler) {
    container.innerHTML = ''; // Clear previous
    for (let i = 0; i < ACTIONS.length; i++) {
        const actionIndex = i;
        const wrapper = document.createElement('div');
        wrapper.className = 'action-btn-wrapper';
        // Add click handler if provided (for Manual Mode)
        if (clickHandler) {
            wrapper.classList.add('clickable');
            wrapper.addEventListener('click', () => clickHandler(actionIndex));
        }
        const canvas = createActionImage(actionIndex, 40); // Use existing function
        wrapper.appendChild(canvas);
        const label = document.createElement('div');
        label.textContent = ACTION_LABELS[actionIndex];
        label.className = 'action-btn-label';
        wrapper.appendChild(label);
        const typeLabel = document.createElement('div');
        typeLabel.textContent = ACTION_TYPES[actionIndex];
        typeLabel.className = 'action-btn-type';
        wrapper.appendChild(typeLabel);
        container.appendChild(wrapper);
    }
}
function updateDemoDisplay(ui) {
    renderGrid(ui.demoTargetCanvas, sharedTargetPattern, '#e29d43ff');
    if (demoPlaybackState && bestSequence) {
        renderGrid(ui.demoBestCanvas, demoPlaybackState, '#4299e1', true, demoAgentX, demoAgentY);
        const matchPercent = calculateMatchPercentage(demoPlaybackState, sharedTargetPattern);
        let statsHTML = `Playback Step: <strong>${demoPlaybackStep}/${bestSequence.length}</strong><br>
                     Temporary Pattern Match: <strong>${matchPercent.toFixed(1)}%</strong>`;
        if (demoPlaybackStep >= bestSequence.length) {
            statsHTML += `<br>Final Fitness Score: <strong>${matchPercent.toFixed(1)}%</strong>`;
        }
        else {
            statsHTML += `<br>Final Fitness Score: <strong>N/A</strong>`;
        }
        ui.demoPlaybackLabel.innerHTML = statsHTML;
        const container = ui.demoActionsCanvas;
        container.innerHTML = '';
        container.style.display = 'flex';
        container.style.flexWrap = 'wrap';
        container.style.gap = '4px';
        container.style.padding = '8px';
        container.style.backgroundColor = '#f7fafc';
        container.style.border = '1px solid #e2e8f0';
        container.style.borderRadius = '0.375rem';
        for (let i = 0; i < bestSequence.length; i++) {
            const wrapper = document.createElement('div');
            wrapper.style.display = 'flex';
            wrapper.style.flexDirection = 'column';
            wrapper.style.alignItems = 'center';
            wrapper.style.gap = '2px';
            if (i === demoPlaybackStep) {
                wrapper.style.backgroundColor = '#71e859ff';
                wrapper.style.padding = '2px';
                wrapper.style.borderRadius = '4px';
            }
            const canvas = createActionImage(bestSequence[i], 40);
            wrapper.appendChild(canvas);
            const label = document.createElement('div');
            label.textContent = ACTION_LABELS[bestSequence[i]];
            label.style.fontSize = '10px';
            label.style.fontWeight = '600';
            label.style.color = '#2d3748';
            wrapper.appendChild(label);
            container.appendChild(wrapper);
        }
    }
    else {
        renderGrid(ui.demoBestCanvas, new Uint8Array(GRID_SIZE * GRID_SIZE), '#4299e1');
    }
}
function demoStepForward(ui) {
    if (!bestSequence || !demoPlaybackState || demoPlaybackStep >= bestSequence.length)
        return;
    const actionIndex = bestSequence[demoPlaybackStep];
    const agentPos = { x: demoAgentX, y: demoAgentY };
    demoPlaybackState = conwayStep(demoPlaybackState);
    demoPlaybackState = applyAction(demoPlaybackState, actionIndex, agentPos);
    demoAgentX = agentPos.x;
    demoAgentY = agentPos.y;
    demoPlaybackStep++;
    updateDemoDisplay(ui);
    if (demoPlaybackStep >= bestSequence.length) {
        log(ui.log, 'Demo sequence finished.');
        if (demoInterval)
            clearInterval(demoInterval);
        demoInterval = null;
        ui.demoPlayBtn.disabled = true;
        ui.demoPauseBtn.disabled = true;
    }
}
function resetDemo(ui) {
    if (demoInterval)
        clearInterval(demoInterval);
    demoInterval = null;
    demoPlaybackState = new Uint8Array(currentInitialState);
    demoPlaybackStep = 0;
    demoAgentX = 5;
    demoAgentY = 5;
    ui.demoPlayBtn.disabled = bestSequence === null;
    ui.demoPauseBtn.disabled = true;
    ui.demoStepBtn.disabled = bestSequence === null;
    updateDemoDisplay(ui);
}
function initializeDemoMode() {
    const ui = getUIElements();
    if (!sharedTargetPattern || !bestSequence) {
        log(ui.log, 'No data for demo. Run Evolve Mode first to generate a sequence.');
        demoPlaybackState = new Uint8Array(GRID_SIZE * GRID_SIZE);
    }
    else {
        log(ui.log, `Loaded sequence with ${bestSequence.length} steps for demo.`);
    }
    ui.demoPlayBtn.onclick = () => {
        if (!bestSequence)
            return;
        if (demoPlaybackStep >= bestSequence.length)
            resetDemo(ui);
        ui.demoPlayBtn.disabled = true;
        ui.demoPauseBtn.disabled = false;
        demoInterval = setInterval(() => demoStepForward(ui), 400);
    };
    ui.demoPauseBtn.onclick = () => {
        if (demoInterval)
            clearInterval(demoInterval);
        demoInterval = null;
        ui.demoPlayBtn.disabled = false;
        ui.demoPauseBtn.disabled = true;
    };
    ui.demoResetBtn.onclick = () => resetDemo(ui);
    ui.demoStepBtn.onclick = () => demoStepForward(ui);
    resetDemo(ui);
}
function initApp() {
    if (sharedTargetPattern) {
        const patternIndices = [
            (4 * GRID_SIZE + 4), (4 * GRID_SIZE + 5), // Top 2x2
            (5 * GRID_SIZE + 4), (5 * GRID_SIZE + 5),
            (5 * GRID_SIZE + 5), (5 * GRID_SIZE + 6), // Bottom-right 2x2 (overlapping at 5,5)
            (6 * GRID_SIZE + 5), (6 * GRID_SIZE + 6)
        ];
        // Use a Set to handle the overlap gracefully
        const uniqueIndices = new Set(patternIndices);
        for (const idx of uniqueIndices) {
            sharedTargetPattern[idx] = 1;
        }
    }
    const ui = getUIElements();
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const target = e.currentTarget;
            currentMode = target.dataset.mode;
            document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
            target.classList.add('active');
            handleModeChange();
        });
    });
    ui.startBtn.addEventListener('click', async () => {
        ui.log.textContent = '';
        try {
            await runEvolution(ui);
        }
        catch (err) {
            log(ui.log, 'FATAL ERROR:', err?.message ?? String(err));
            isRunning = false;
            ui.startBtn.disabled = false;
            ui.stopBtn.disabled = true;
        }
    });
    ui.stopBtn.addEventListener('click', () => {
        isRunning = false;
        log(ui.log, 'Evolution stopped by user.');
    });
    document.addEventListener('keydown', handleManualKeyDown);
    const targetClickHandler = (e) => {
        const canvas = e.currentTarget;
        if (!sharedTargetPattern)
            sharedTargetPattern = new Uint8Array(GRID_SIZE * GRID_SIZE);
        const rect = canvas.getBoundingClientRect();
        // Use display size (rect) instead of canvas buffer size to handle click position after window is resized
        const displayWidth = rect.width;
        // const displayHeight = rect.height;
        const cell = displayWidth / GRID_SIZE;
        // const cell = canvas.width / GRID_SIZE;
        const x = Math.floor((e.clientX - rect.left) / cell);
        const y = Math.floor((e.clientY - rect.top) / cell);
        if (x >= 0 && x < GRID_SIZE && y >= 0 && y < GRID_SIZE) {
            const idx = y * GRID_SIZE + x;
            sharedTargetPattern[idx] = 1 - sharedTargetPattern[idx];
            renderGrid(canvas, sharedTargetPattern, '#e29d43ff');
        }
    };
    ui.targetVizCanvas.addEventListener('click', targetClickHandler);
    ui.demoTargetCanvas.addEventListener('click', targetClickHandler);
    initManualMode(ui);
    handleModeChange(); // This will render the initial pattern
    log(ui.log, "App ready. Default target pattern loaded. Press 'Start Evolution' or go to Manual mode.");
    log(ui.log, 'Simulation Note: Action Sequence Fitness evaluation computed with a WebGPU shader. Make sure your browser is WebGPU-compatible and enabled.');
    log(ui.log, 'RL Gymnasium for training your own Reinforcement Agent 🤖🏋🏻: https://github.com/Digital-Physics/game-of-life-pattern-RL-gym');
}
window.addEventListener('DOMContentLoaded', initApp);
