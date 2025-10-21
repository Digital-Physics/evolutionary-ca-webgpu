// src/app.ts
// WebGPU CA compute + GPU fitness + genetic algorithm
// Modes: manual, evolve, demo

// --- TYPE DEFINITIONS ---
type UIElements = {
  population: HTMLInputElement;
  steps: HTMLInputElement;
  generations: HTMLInputElement;
  elite: HTMLInputElement;
  mut: HTMLInputElement;
  vizFreq: HTMLInputElement;
  startBtn: HTMLButtonElement;
  stopBtn: HTMLButtonElement;
  log: HTMLElement;
  clearPatternBtn: HTMLButtonElement;
  savePatternBtn: HTMLButtonElement;
  statsDiv: HTMLElement;
  manualDemoCanvas: HTMLCanvasElement;
  manualTargetCanvas: HTMLCanvasElement;
  manualStats: HTMLElement;
  manualMaxStepsLabel: HTMLElement;
  demoTargetCanvas: HTMLCanvasElement;
  demoBestCanvas: HTMLCanvasElement;
  demoActionsCanvas: HTMLCanvasElement;
  demoStep: HTMLElement;
  demoMaxSteps: HTMLElement;
  demoPlayBtn: HTMLButtonElement;
  demoPauseBtn: HTMLButtonElement;
  demoResetBtn: HTMLButtonElement;
  demoStepBtn: HTMLButtonElement;
  evolveBestCanvas: HTMLCanvasElement;
  evolveCurrentCanvas: HTMLCanvasElement;
  targetVizCanvas: HTMLCanvasElement;
  manualResetBtn: HTMLButtonElement;
  fitnessDistCanvas: HTMLCanvasElement;
  diversityCanvas: HTMLCanvasElement;
};

// --- CONSTANTS AND CONFIGURATION ---
const GRID_SIZE = 12;
const MANUAL_MAX_STEPS = 12;
const ACTIONS = ['up', 'down', 'left', 'right', 'do_nothing', 'write_0000', 'write_0001', 'write_0010', 'write_0011', 'write_0100', 'write_0101', 'write_0110', 'write_0111', 'write_1000', 'write_1001', 'write_1010', 'write_1011', 'write_1100', 'write_1101', 'write_1110', 'write_1111'];
// Action decoder translates a numeric index to an action object
const ACTION_DECODER = ACTIONS.map((name, index) => {
  const parts = name.split('_');
  if (parts.length === 2 && parts[0] === 'write') {
    const bitString = parts[1];
    // For a pattern "abcd", bits are [a, b, c, d]
    return { type: 'write', patBits: bitString.split('').map(b => parseInt(b, 2)) as [number, number, number, number], index };
  }
  return { type: name, patBits: [0, 0, 0, 0] as [number, number, number, number], index };
});

// --- GLOBAL STATE ---
let currentMode: 'evolve' | 'manual' | 'demo' = 'evolve';

// Manual Mode State
let manualDemoState = new Uint8Array(GRID_SIZE * GRID_SIZE);
let manualTargetState = new Uint8Array(GRID_SIZE * GRID_SIZE);
let manualStep = 0;
let manualAgentX = GRID_SIZE >> 1;
let manualAgentY = GRID_SIZE >> 1;

// Shared state between modes
let sharedTargetPattern: Uint8Array | null = null;

// Evolve Mode State
let isRunning = false;
let bestFitness = 0;
let bestSequence: Uint32Array | null = null;
const currentInitialState = new Uint8Array(GRID_SIZE * GRID_SIZE); // Start from an empty grid

// Demo Mode State
let demoPlaybackState: Uint8Array | null = null;
let demoPlaybackStep = 0;
let demoInterval: number | null = null;
let demoAgentX = GRID_SIZE >> 1;
let demoAgentY = GRID_SIZE >> 1;

// WebGPU Context
let gpuDevice: GPUDevice | null = null;
let pipeline: GPUComputePipeline | null = null;

// Statistics for charts
let avgFitnessHistory: number[] = [];
let allTimeBestFitnessHistory: number[] = [];
let diversityHistory: number[] = [];

// --- COMPUTE SHADER (WGSL) ---
// The shader is kept embedded in the script for simplicity and portability of a single-file app.
// It simulates the cellular automata and calculates fitness on the GPU.
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

// Helper to get 1D index from 2D coordinates
fn get_idx(x: u32, y: u32) -> u32 {
  return y * params.gridSize + x;
}

// Helper for modular arithmetic with proper wrapping for negative numbers
fn wrap_add(a: i32, b: i32, m: i32) -> i32 {
  let v = (a + b) % m;
  if (v < 0) { return v + m; }
  return v;
}

// --- Action Application ---
// Moves the agent's coordinates within the state representation
fn apply_action_move(state: ptr<function, array<u32, 144>>, action_index: u32) {
  var ax = (*state)[140u]; // Agent X is stored at index 140
  var ay = (*state)[141u]; // Agent Y is stored at index 141
  let grid_size = params.gridSize;

  if (action_index == 0u) { ay = (ay + grid_size - 1u) % grid_size; } // Up
  else if (action_index == 1u) { ay = (ay + 1u) % grid_size; }      // Down
  else if (action_index == 2u) { ax = (ax + grid_size - 1u) % grid_size; } // Left
  else if (action_index == 3u) { ax = (ax + 1u) % grid_size; }      // Right
  
  (*state)[140u] = ax;
  (*state)[141u] = ay;
}

// Writes a 2x2 pattern to the grid at the agent's location
fn apply_action_write(state: ptr<function, array<u32, 144>>, action_index: u32) {
  let pat = action_index - 5u;
  let ax = (*state)[140u];
  let ay = (*state)[141u];
  let grid_size_i32 = i32(params.gridSize);

  for (var i: i32 = 0; i < 2; i = i + 1) { // y offset
    for (var j: i32 = 0; j < 2; j = j + 1) { // x offset
      // FIX: Bit order corrected to match JS implementation for consistent simulation.
      // Unpacks bits from left-to-right (MSB to LSB).
      let bit_index = 3u - u32(i * 2 + j);
      let bit = (pat >> bit_index) & 1u;
      
      let wx = u32(wrap_add(i32(ax) + j, 0, grid_size_i32));
      let wy = u32(wrap_add(i32(ay) + i, 0, grid_size_i32));
      (*state)[get_idx(wx, wy)] = bit;
    }
  }
}

// --- Conway's Game of Life Simulation Step ---
fn conway_step(state: ptr<function, array<u32, 144>>) {
  var next_state: array<u32, 144>;
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

  // Copy next state back to current state
  for (var i: u32 = 0; i < grid_size * grid_size; i = i + 1) {
    (*state)[i] = next_state[i];
  }
}

// --- MAIN COMPUTE FUNCTION ---
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx >= params.batchSize) { return; }

  let state_size = params.gridSize * params.gridSize;
  let state_offset = idx * state_size;

  // Load initial state for this invocation
  var current_state: array<u32, 144>;
  for (var i: u32 = 0; i < state_size; i = i + 1) {
    current_state[i] = inputStates[state_offset + i];
  }
  // Initialize agent position in the center
  current_state[140u] = params.gridSize / 2u;
  current_state[141u] = params.gridSize / 2u;

  // --- Main Simulation Loop ---
  let sequence_offset = idx * params.steps;
  for (var s: u32 = 0; s < params.steps; s = s + 1) {
    let action_index = inputSequences[sequence_offset + s];

    // 1. Apply user action (move, write, or nothing)
    if (action_index <= 3u) {
        apply_action_move(&current_state, action_index);
    } else if (action_index >= 5u) {
        apply_action_write(&current_state, action_index);
    }
    
    // 2. Evolve the grid by one step of Conway's Game of Life
    conway_step(&current_state);
  }

  // --- Fitness Calculation ---
  var matches: u32 = 0u;
  for (var i2: u32 = 0; i2 < state_size; i2 = i2 + 1) {
    if (current_state[i2] == targetPattern[i2]) {
      matches = matches + 1u;
    }
  }

  // Fitness is the percentage of matching cells (0-100)
  fitness[idx] = (matches * 100u) / state_size;
  
  // Write final state to output buffer
  for (var i3: u32 = 0; i3 < state_size; i3 = i3 + 1) {
    outputStates[state_offset + i3] = current_state[i3];
  }
}
`;

// --- CPU-SIDE SIMULATION (for Manual/Demo modes) ---
function conwayStep(grid: Uint8Array): Uint8Array {
  const size = Math.sqrt(grid.length);
  const nextGrid = new Uint8Array(grid.length);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      let neighbors = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          const nx = (x + dx + size) % size;
          const ny = (y + dy + size) % size;
          neighbors += grid[ny * size + nx];
        }
      }
      const current = grid[y * size + x];
      if (current === 1) {
        if (neighbors === 2 || neighbors === 3) nextGrid[y * size + x] = 1;
      } else {
        if (neighbors === 3) nextGrid[y * size + x] = 1;
      }
    }
  }
  return nextGrid;
}

function applyAction(state: Uint8Array, actionIndex: number, agentPos: {x: number, y: number}): Uint8Array {
  const size = Math.sqrt(state.length);
  const newState = new Uint8Array(state);
  const action = ACTION_DECODER[actionIndex];

  switch (action.type) {
    case 'up': agentPos.y = (agentPos.y - 1 + size) % size; break;
    case 'down': agentPos.y = (agentPos.y + 1) % size; break;
    case 'left': agentPos.x = (agentPos.x - 1 + size) % size; break;
    case 'right': agentPos.x = (agentPos.x + 1) % size; break;
    case 'write':
      for (let i = 0; i < 2; i++) { // y offset
        for (let j = 0; j < 2; j++) { // x offset
          const yy = (agentPos.y + i + size) % size;
          const xx = (agentPos.x + j + size) % size;
          newState[yy * size + xx] = action.patBits[i * 2 + j];
        }
      }
      break;
  }
  return newState;
}

function evaluateSequenceToGrid(seq: Uint32Array | null): Uint8Array | null {
    if (!seq) return null;
    const size = GRID_SIZE;
    let state = new Uint8Array(currentInitialState);
    let agent = { x: size >> 1, y: size >> 1 };
    for (const actionIndex of seq) {
        state = applyAction(state, actionIndex, agent) as any;
        state = conwayStep(state) as any;
    }
    return state;
}


// --- RENDERING ---
function renderGrid(canvas: HTMLCanvasElement, grid: Uint8Array | null, cellColor: string = '#4299e1', showAgent: boolean = false, agentX: number = 0, agentY: number = 0) {
  const size = GRID_SIZE;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

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

function getUIElements(): UIElements {
  return {
    population: document.getElementById('population') as HTMLInputElement,
    steps: document.getElementById('steps') as HTMLInputElement,
    generations: document.getElementById('generations') as HTMLInputElement,
    elite: document.getElementById('elite') as HTMLInputElement,
    mut: document.getElementById('mut') as HTMLInputElement,
    vizFreq: document.getElementById('vizFreq') as HTMLInputElement,
    startBtn: document.getElementById('startBtn') as HTMLButtonElement,
    stopBtn: document.getElementById('stopBtn') as HTMLButtonElement,
    statsDiv: document.getElementById('statsDiv') as HTMLElement,
    manualDemoCanvas: document.getElementById('manualDemoCanvas') as HTMLCanvasElement,
    manualTargetCanvas: document.getElementById('manualTargetCanvas') as HTMLCanvasElement,
    manualStats: document.getElementById('manualStats') as HTMLElement,
    manualMaxStepsLabel: document.getElementById('manualMaxStepsLabel') as HTMLElement,
    clearPatternBtn: document.getElementById('clearPatternBtn') as HTMLButtonElement,
    savePatternBtn: document.getElementById('savePatternBtn') as HTMLButtonElement,
    demoTargetCanvas: document.getElementById('demoTargetCanvas') as HTMLCanvasElement,
    demoBestCanvas: document.getElementById('demoBestCanvas') as HTMLCanvasElement,
    demoActionsCanvas: document.getElementById('demoActionsCanvas') as HTMLCanvasElement,
    demoStep: document.getElementById('demoStep') as HTMLElement,
    demoMaxSteps: document.getElementById('demoMaxSteps') as HTMLElement,
    demoPlayBtn: document.getElementById('demoPlayBtn') as HTMLButtonElement,
    demoPauseBtn: document.getElementById('demoPauseBtn') as HTMLButtonElement,
    demoResetBtn: document.getElementById('demoResetBtn') as HTMLButtonElement,
    demoStepBtn: document.getElementById('demoStepBtn') as HTMLButtonElement,
    log: document.getElementById('log') as HTMLElement,
    evolveBestCanvas: document.getElementById('evolveBestCanvas') as HTMLCanvasElement,
    evolveCurrentCanvas: document.getElementById('evolveCurrentCanvas') as HTMLCanvasElement,
    targetVizCanvas: document.getElementById('targetVizCanvas') as HTMLCanvasElement,
    manualResetBtn: document.getElementById('manualResetBtn') as HTMLButtonElement,
    fitnessDistCanvas: document.getElementById('fitnessDistCanvas') as HTMLCanvasElement,
    diversityCanvas: document.getElementById('diversityCanvas') as HTMLCanvasElement,
  };
}

// --- MODE HANDLING ---
function handleModeChange() {
  document.querySelectorAll('.mode-content').forEach(el => (el as HTMLElement).style.display = 'none');
  const selectedModeEl = document.getElementById(`${currentMode}-mode`);
  if (selectedModeEl) {
    selectedModeEl.style.display = 'block';
  }

  const ui = getUIElements();
  if (currentMode === 'manual') {
    renderGrid(ui.manualTargetCanvas, sharedTargetPattern, '#c53030');
    updateManualDisplay();
  } else if (currentMode === 'evolve') {
    renderGrid(ui.targetVizCanvas, sharedTargetPattern, '#c53030');
  } else if (currentMode === 'demo') {
    initializeDemoMode();
  }
}

// --- MANUAL MODE ---
function updateManualDisplay() {
  const ui = getUIElements();
  renderGrid(ui.manualDemoCanvas, manualDemoState, '#4299e1', true, manualAgentX, manualAgentY);
  renderGrid(ui.manualTargetCanvas, manualTargetState, '#c53030');
  ui.manualStats.innerHTML = `
    Step: <strong>${manualStep}/${MANUAL_MAX_STEPS}</strong><br>
    Agent Position: <strong>(${manualAgentX}, ${manualAgentY})</strong>
  `;
}

function initManualMode(ui: UIElements) {
    ui.manualMaxStepsLabel.textContent = MANUAL_MAX_STEPS.toString();
    
    const targetClickHandler = (e: MouseEvent) => {
        const canvas = e.currentTarget as HTMLCanvasElement;
        const rect = canvas.getBoundingClientRect();
        const cell = canvas.width / GRID_SIZE;
        const x = Math.floor((e.clientX - rect.left) / cell);
        const y = Math.floor((e.clientY - rect.top) / cell);
        if (x >= 0 && x < GRID_SIZE && y >= 0 && y < GRID_SIZE) {
            const idx = y * GRID_SIZE + x;
            manualTargetState[idx] = 1 - manualTargetState[idx];
            sharedTargetPattern = new Uint8Array(manualTargetState);
            updateManualDisplay();
        }
    };
    ui.manualTargetCanvas.addEventListener('click', targetClickHandler);

    ui.savePatternBtn.addEventListener('click', () => {
        manualTargetState = new Uint8Array(manualDemoState);
        sharedTargetPattern = new Uint8Array(manualTargetState);
        log(ui.log, 'Saved manual canvas to shared target pattern.');
        updateManualDisplay();
    });

    ui.clearPatternBtn.addEventListener('click', () => {
        manualTargetState.fill(0);
        sharedTargetPattern = new Uint8Array(manualTargetState);
        updateManualDisplay();
    });

    ui.manualResetBtn.addEventListener('click', () => {
        manualDemoState.fill(0);
        manualStep = 0;
        manualAgentX = GRID_SIZE >> 1;
        manualAgentY = GRID_SIZE >> 1;
        updateManualDisplay();
        log(ui.log, 'Manual mode reset.');
    });
    
    updateManualDisplay();
}

function handleManualKeyDown(e: KeyboardEvent) {
    if (currentMode !== 'manual' || manualStep >= MANUAL_MAX_STEPS) return;

    const key = e.key.toLowerCase();
    let actionTaken = false;
    let actionIndex = -1;

    if (key.startsWith('arrow')) {
        actionIndex = ['arrowup', 'arrowdown', 'arrowleft', 'arrowright'].indexOf(key);
    } else if (key === ' ') {
        actionIndex = 4; // do_nothing
    } else if (/^[0-9a-f]$/.test(key)) {
        const patternNum = parseInt(key, 16);
        actionIndex = 5 + 16 - 1 - patternNum; // Find the correct write action
    }

    if (actionIndex !== -1) {
        e.preventDefault();
        const agentPos = { x: manualAgentX, y: manualAgentY };
        manualDemoState = applyAction(manualDemoState, actionIndex, agentPos) as any;
        manualAgentX = agentPos.x;
        manualAgentY = agentPos.y;

        // An action is always followed by a CA step
        manualDemoState = conwayStep(manualDemoState) as any;
        manualStep++;
        actionTaken = true;
    } else if (key === 's') {
        e.preventDefault();
        getUIElements().savePatternBtn.click();
    }

    if (actionTaken) {
        updateManualDisplay();
    }
}


// --- LOGGING ---
function log(el: HTMLElement | null, ...args: any[]) {
  if (!el) return;
  el.textContent += args.join(' ') + '\n';
  el.scrollTop = el.scrollHeight;
}

// --- WEBGPU SETUP ---
async function initWebGPU(): Promise<GPUDevice> {
  if (gpuDevice) return gpuDevice;
  if (!navigator.gpu) throw new Error('WebGPU not supported on this browser.');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No compatible GPUAdapter found.');
  gpuDevice = await adapter.requestDevice();
  return gpuDevice;
}

async function setupComputePipeline() {
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

// --- EVOLUTION MODE ---
async function runEvolution(ui: UIElements) {
  if (!sharedTargetPattern || sharedTargetPattern.every(cell => cell === 0)) {
    log(ui.log, 'ERROR: No target pattern set. Please draw a pattern on the target canvas first.');
    return;
  }

  isRunning = true;
  ui.startBtn.disabled = true;
  ui.stopBtn.disabled = false;
  
  const device = await initWebGPU();
  await setupComputePipeline();

  // Get parameters from UI
  const batchSize = Number(ui.population.value);
  const steps = Number(ui.steps.value);
  const generations = Number(ui.generations.value);
  const mutationRate = Number(ui.mut.value);
  const eliteFrac = Number(ui.elite.value);
  const vizFreq = Number(ui.vizFreq.value);

  // Prepare GPU buffers
  const stateSize = GRID_SIZE * GRID_SIZE;
  const stateBufferSize = batchSize * stateSize * 4;
  const seqBufferSize = batchSize * steps * 4;

  const inputStatesBuffer = device.createBuffer({ size: stateBufferSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const inputSequencesBuffer = device.createBuffer({ size: seqBufferSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const outputStatesBuffer = device.createBuffer({ size: stateBufferSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const targetBuffer = device.createBuffer({ size: stateSize * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const fitnessBuffer = device.createBuffer({ size: batchSize * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const fitnessReadBuffer = device.createBuffer({ size: batchSize * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const paramsBuffer = device.createBuffer({ size: 3 * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  // Write initial data to buffers
  const initialStates = new Uint32Array(batchSize * stateSize);
  // All simulations start from an empty grid
  device.queue.writeBuffer(inputStatesBuffer, 0, initialStates);

  const targetPattern32 = new Uint32Array(sharedTargetPattern);
  device.queue.writeBuffer(targetBuffer, 0, targetPattern32);
  
  const paramsArray = new Uint32Array([GRID_SIZE, batchSize, steps]);
  device.queue.writeBuffer(paramsBuffer, 0, paramsArray);

  const bindGroup = device.createBindGroup({
    layout: pipeline!.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: paramsBuffer } },
      { binding: 1, resource: { buffer: inputStatesBuffer } },
      { binding: 2, resource: { buffer: inputSequencesBuffer } },
      { binding: 3, resource: { buffer: outputStatesBuffer } },
      { binding: 4, resource: { buffer: targetBuffer } },
      { binding: 5, resource: { buffer: fitnessBuffer } },
    ],
  });

  // Initialize population
  let populationSequences = new Uint32Array(batchSize * steps).map(() => Math.floor(Math.random() * ACTIONS.length));
  
  // Reset stats
  bestFitness = 0;
  bestSequence = null;
  avgFitnessHistory = [];
  allTimeBestFitnessHistory = [];
  diversityHistory = [];

  // Evolution loop
  for (let gen = 0; gen < generations && isRunning; gen++) {
    device.queue.writeBuffer(inputSequencesBuffer, 0, populationSequences);

    const commandEncoder = device.createCommandEncoder();
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(pipeline!);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(batchSize / 64));
    pass.end();
    commandEncoder.copyBufferToBuffer(fitnessBuffer, 0, fitnessReadBuffer, 0, batchSize * 4);
    device.queue.submit([commandEncoder.finish()]);

    await fitnessReadBuffer.mapAsync(GPUMapMode.READ);
    const fitnessArray = new Uint32Array(fitnessReadBuffer.getMappedRange().slice(0));
    fitnessReadBuffer.unmap();

    let maxFit = 0;
    let maxIdx = -1;
    let sumFitness = 0;
    for (let i = 0; i < batchSize; i++) {
        sumFitness += fitnessArray[i];
        if (fitnessArray[i] > maxFit) {
            maxFit = fitnessArray[i];
            maxIdx = i;
        }
    }
    const avgFitness = sumFitness / batchSize;

    if (maxFit > bestFitness) {
      bestFitness = maxFit;
      bestSequence = populationSequences.slice(maxIdx * steps, (maxIdx + 1) * steps);
      log(ui.log, `Gen ${gen}: New best fitness ${bestFitness}%`);
    }
    
    // Calculate diversity
    const seqSet = new Set();
    for(let i = 0; i < batchSize; i++) {
        seqSet.add(populationSequences.slice(i * steps, (i + 1) * steps).join(','));
    }
    diversityHistory.push(seqSet.size / batchSize);
    avgFitnessHistory.push(avgFitness);
    allTimeBestFitnessHistory.push(bestFitness);

    // Visualization & Stats update
    if (gen % vizFreq === 0 || gen === generations - 1 || bestFitness === 100) {
      const bestSeqStr = bestSequence ? Array.from(bestSequence).map(a => ACTIONS[a]).join(', ') : 'N/A';
      ui.statsDiv.innerHTML = `
        Gen: <strong>${gen} / ${generations}</strong><br>
        Best Fitness: <strong>${bestFitness}%</strong><br>
        <hr style="margin: 4px 0; border-top: 1px solid #e2e8f0;">
        Best Sequence: <small>${bestSeqStr}</small>
      `;
      renderGrid(ui.evolveBestCanvas, evaluateSequenceToGrid(bestSequence), '#38a169');
      renderGrid(ui.targetVizCanvas, sharedTargetPattern, '#c53030');
      
      renderFitnessChart(ui.fitnessDistCanvas, avgFitnessHistory, allTimeBestFitnessHistory);
      renderDiversityChart(ui.diversityCanvas, diversityHistory);
      await new Promise(r => setTimeout(r, 1));
    }

    if (bestFitness === 100) {
      log(ui.log, `Perfect match found in generation ${gen}!`);
      break;
    }

    // --- Genetic Algorithm: Selection, Crossover, Mutation ---
    const sortedIndices = Array.from({ length: batchSize }, (_, i) => i).sort((a, b) => fitnessArray[b] - fitnessArray[a]);
    const newPopulation = new Uint32Array(batchSize * steps);
    const eliteCount = Math.floor(batchSize * eliteFrac);
    
    // Elitism: Copy the best individuals directly
    for (let i = 0; i < eliteCount; i++) {
        const bestIdx = sortedIndices[i];
        newPopulation.set(populationSequences.slice(bestIdx * steps, (bestIdx + 1) * steps), i * steps);
    }

    // Crossover and Mutation for the rest
    for (let i = eliteCount; i < batchSize; i++) {
        const parentA_idx = sortedIndices[Math.floor(Math.random() * batchSize * 0.5)]; // Tournament selection from top 50%
        const parentB_idx = sortedIndices[Math.floor(Math.random() * batchSize * 0.5)];
        const parentA = populationSequences.slice(parentA_idx * steps, (parentA_idx + 1) * steps);
        const parentB = populationSequences.slice(parentB_idx * steps, (parentB_idx + 1) * steps);
        const crossPoint = Math.floor(Math.random() * steps);
        const child = new Uint32Array(steps);
        for (let s = 0; s < steps; s++) {
            child[s] = (s < crossPoint) ? parentA[s] : parentB[s];
            if (Math.random() < mutationRate) {
                child[s] = Math.floor(Math.random() * ACTIONS.length);
            }
        }
        newPopulation.set(child, i * steps);
    }
    populationSequences = newPopulation;
  }

  isRunning = false;
  ui.startBtn.disabled = false;
  ui.stopBtn.disabled = true;
  log(ui.log, 'Evolution complete.');
}


// --- CHART RENDERING ---
function renderChart(canvas: HTMLCanvasElement, datasets: {data: number[], color: string, label: string}[], yLabel: string, xLabel: string, yMax: number) {
    const ctx = canvas.getContext('2d');
    if (!ctx || datasets.length === 0 || datasets[0].data.length === 0) return;

    const { width, height } = canvas;
    const p = { t: 30, r: 20, b: 40, l: 50 }; // padding

    ctx.clearRect(0, 0, width, height);
    ctx.font = '12px sans-serif';

    const xRange = width - p.l - p.r;
    const yRange = height - p.t - p.b;
    const numPoints = datasets[0].data.length;

    // Draw grid lines and Y-axis labels
    ctx.strokeStyle = '#e2e8f0';
    ctx.fillStyle = '#718096';
    ctx.lineWidth = 1;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let i = 0; i <= 5; i++) {
        const y = p.t + yRange * (1 - i / 5);
        ctx.beginPath();
        ctx.moveTo(p.l, y);
        ctx.lineTo(p.l + xRange, y);
        ctx.stroke();
        ctx.fillText((yMax * i / 5).toFixed(yMax < 2 ? 1 : 0), p.l - 8, y);
    }

    // Draw data lines
    datasets.forEach(({ data, color }) => {
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let i = 0; i < numPoints; i++) {
            const x = p.l + (xRange * i) / Math.max(1, numPoints - 1);
            const y = p.t + yRange * (1 - data[i] / yMax);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
    });

    // Draw axis labels
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

function renderFitnessChart(canvas: HTMLCanvasElement, avg: number[], best: number[]) {
    renderChart(canvas, [
        { data: best, color: '#38a169', label: 'Best Fitness' },
        { data: avg, color: '#4299e1', label: 'Avg Fitness' }
    ], 'Fitness %', 'Generation', 100);
}

function renderDiversityChart(canvas: HTMLCanvasElement, diversity: number[]) {
    renderChart(canvas, [
        { data: diversity, color: '#9f7aea', label: 'Diversity' }
    ], 'Diversity Ratio', 'Generation', 1);
}


// --- DEMO MODE ---
function updateDemoDisplay(ui: UIElements) {
  renderGrid(ui.demoTargetCanvas, sharedTargetPattern, '#c53030');

  if (demoPlaybackState && bestSequence) {
    renderGrid(ui.demoBestCanvas, demoPlaybackState, '#4299e1', true, demoAgentX, demoAgentY);
    ui.demoStep.textContent = demoPlaybackStep.toString();

    const ctx = ui.demoActionsCanvas.getContext('2d');
    if (ctx) {
      const { width, height } = ui.demoActionsCanvas;
      const itemW = width / bestSequence.length;
      ctx.clearRect(0, 0, width, height);
      
      const actionLabels = ['↑','↓','←','→','∅', ...Array.from({length:16}, (_,i)=>(15-i).toString(16).toUpperCase())];

      for (let i = 0; i < bestSequence.length; i++) {
        if (i === demoPlaybackStep) ctx.fillStyle = '#f6ad55';
        else if (i < demoPlaybackStep) ctx.fillStyle = '#f7fafc';
        else ctx.fillStyle = '#ffffff';
        
        ctx.fillRect(i * itemW, 0, itemW, height);
        ctx.strokeStyle = '#cbd5e0';
        ctx.strokeRect(i * itemW, 0, itemW, height);
        
        ctx.fillStyle = '#2d3748';
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(actionLabels[bestSequence[i]] || '?', i * itemW + itemW / 2, height / 2);
      }
    }
  } else {
    renderGrid(ui.demoBestCanvas, new Uint8Array(GRID_SIZE*GRID_SIZE), '#4299e1');
  }
}

function demoStepForward(ui: UIElements) {
  if (!bestSequence || !demoPlaybackState || demoPlaybackStep >= bestSequence.length) return;

  const actionIndex = bestSequence[demoPlaybackStep];
  const agentPos = { x: demoAgentX, y: demoAgentY };
  demoPlaybackState = applyAction(demoPlaybackState, actionIndex, agentPos);
  demoAgentX = agentPos.x;
  demoAgentY = agentPos.y;
  demoPlaybackState = conwayStep(demoPlaybackState);

  demoPlaybackStep++;
  updateDemoDisplay(ui);

  if (demoPlaybackStep >= bestSequence.length) {
    log(ui.log, 'Demo sequence finished.');
    if (demoInterval) clearInterval(demoInterval);
    demoInterval = null;
    ui.demoPlayBtn.disabled = true;
    ui.demoPauseBtn.disabled = true;
  }
}

function resetDemo(ui: UIElements) {
  if (demoInterval) clearInterval(demoInterval);
  demoInterval = null;
  
  demoPlaybackState = new Uint8Array(currentInitialState);
  demoPlaybackStep = 0;
  demoAgentX = GRID_SIZE >> 1;
  demoAgentY = GRID_SIZE >> 1;

  ui.demoStep.textContent = '0';
  ui.demoPlayBtn.disabled = bestSequence === null;
  ui.demoPauseBtn.disabled = true;
  ui.demoStepBtn.disabled = bestSequence === null;

  updateDemoDisplay(ui);
}

function initializeDemoMode() {
  const ui = getUIElements();
  if (!sharedTargetPattern || !bestSequence) {
    log(ui.log, 'No data for demo. Run Evolve Mode first to generate a sequence.');
    demoPlaybackState = new Uint8Array(GRID_SIZE*GRID_SIZE);
  } else {
    log(ui.log, `Loaded sequence with ${bestSequence.length} steps for demo.`);
    ui.demoMaxSteps.textContent = bestSequence.length.toString();
  }

  ui.demoPlayBtn.onclick = () => {
    if (!bestSequence) return;
    if (demoPlaybackStep >= bestSequence.length) resetDemo(ui);
    ui.demoPlayBtn.disabled = true;
    ui.demoPauseBtn.disabled = false;
    demoInterval = setInterval(() => demoStepForward(ui), 400);
  };
  ui.demoPauseBtn.onclick = () => {
    if (demoInterval) clearInterval(demoInterval);
    demoInterval = null;
    ui.demoPlayBtn.disabled = false;
    ui.demoPauseBtn.disabled = true;
  };
  ui.demoResetBtn.onclick = () => resetDemo(ui);
  ui.demoStepBtn.onclick = () => demoStepForward(ui);

  resetDemo(ui);
}

// --- APP INITIALIZATION ---
function initApp() {
  const ui = getUIElements();

  // Mode switching
  document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
        const target = e.currentTarget as HTMLElement;
        currentMode = target.dataset.mode as typeof currentMode;
        document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
        target.classList.add('active');
        handleModeChange();
    });
  });

  // Global controls
  ui.startBtn.addEventListener('click', async () => {
    ui.log.textContent = '';
    try {
      await runEvolution(ui);
    } catch (err: any) {
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
  
  // FIX: Attach keydown listener once to the document to prevent multiple bindings.
  document.addEventListener('keydown', handleManualKeyDown);

  // Target canvas editing
  const targetClickHandler = (e: MouseEvent) => {
    const canvas = e.currentTarget as HTMLCanvasElement;
    if (!sharedTargetPattern) sharedTargetPattern = new Uint8Array(GRID_SIZE * GRID_SIZE);
    const rect = canvas.getBoundingClientRect();
    const cell = canvas.width / GRID_SIZE;
    const x = Math.floor((e.clientX - rect.left) / cell);
    const y = Math.floor((e.clientY - rect.top) / cell);
    
    const idx = y * GRID_SIZE + x;
    sharedTargetPattern[idx] = 1 - sharedTargetPattern[idx];
    renderGrid(canvas, sharedTargetPattern, '#c53030');
  };
  ui.targetVizCanvas.addEventListener('click', targetClickHandler);
  ui.demoTargetCanvas.addEventListener('click', targetClickHandler);

  initManualMode(ui);
  handleModeChange();
  log(ui.log, 'App ready. Draw a target pattern, then start evolution.');
}

// Start the application once the DOM is loaded
window.addEventListener('DOMContentLoaded', initApp);


