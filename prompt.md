Can you help me update my website code? I tried to port it over from python to the web using a ts and a webgpu compute shader. It's almost there, but there are still some issues.

The save target pattern functionality in Manual Mode does not seem to update the target pattern canvas before the Evolve mode is selected, only after Start Evolution is clicked.

The Evolve Mode doesn't seem to be working correctly. It's not finding solutions well, and the best pattern and generation sample are often blank. Please make sure the compute shader is set up to test a sequence of actions which involve interleaving actions (move, do nothing(but let Game of Life update), write a 2x2 pattern)

The Manual Mode is also not working properly. The game should freeze when the steps are reached, but the user should be able to reset the game with a button. The step count and iteration isn't working right. The reset should be to blank Game of Life canvas.

The stats aren't the same as python and the visualizations aren't there, especially in Evolve Mode.

Demo Mode isn't showing the action sequence symbols. Also, resetting Demo mode should clear the Game Of Life grid and reset the 2x2 agent square to the middle (squares 6 and 7), just like manual mode.

It would be nice if the log wasn't called the "Evolution Log" but just the "Log" and it was a constant at the bottom of the screen, in every mode, just like the title and the buttons are a constant across every mode.

Make sure the 2x2 window is directly over the squares that can be updated, not shifted to the left and up. Also make sure the grids are initialized with lines, not completely blank.

Please update the app.ts file and index.html file.

// src/app.ts
// WebGPU CA compute + GPU fitness + genetic algorithm
// Modes: manual, evolve, demo

type UIElements = {
  mode: HTMLSelectElement;
  population: HTMLInputElement;
  steps: HTMLInputElement;
  generations: HTMLInputElement;
  elite: HTMLInputElement;
  mut: HTMLInputElement;
  vizFreq: HTMLInputElement;
  startBtn: HTMLButtonElement;
  stopBtn: HTMLButtonElement;
  canvas: HTMLCanvasElement;
  fitnessCanvas: HTMLCanvasElement;
  log: HTMLElement;
  patternCanvas: HTMLCanvasElement;
  clearPatternBtn: HTMLButtonElement;
  savePatternBtn: HTMLButtonElement;
  statsDiv: HTMLElement;
  seqStrip: HTMLCanvasElement;
  manualMode: HTMLElement;
  manualDemoCanvas: HTMLCanvasElement;
  manualTargetCanvas: HTMLCanvasElement;
  manualStats: HTMLElement;
  vizToggle: HTMLInputElement;
  demoMode: HTMLElement;
  demoTargetCanvas: HTMLCanvasElement;
  demoBestCanvas: HTMLCanvasElement;
  demoActionsCanvas: HTMLCanvasElement;
  demoStep: HTMLElement;
  demoMaxSteps: HTMLElement;
  demoPlayBtn: HTMLButtonElement;
  demoPauseBtn: HTMLButtonElement;
  demoResetBtn: HTMLButtonElement;
  demoStepBtn: HTMLButtonElement;
  demoInfo: HTMLElement;
  evolveBestCanvas: HTMLCanvasElement;
  evolveCurrentCanvas: HTMLCanvasElement;
  targetVizCanvas: HTMLCanvasElement;
  evolveMode: HTMLElement;
};

const GRID_SIZE = 12;
const MAX_GENERATIONS_DEFAULT = 200;
const MANUAL_MAX_STEPS = 10;
const ACTIONS = ['up', 'down', 'left', 'right', 'do_nothing', 'write_0000', 'write_0001', 'write_0010', 'write_0011', 'write_0100', 'write_0101', 'write_0110', 'write_0111', 'write_1000', 'write_1001', 'write_1010', 'write_1011', 'write_1100', 'write_1101', 'write_1110', 'write_1111'];
const ACTION_DECODER = ACTIONS.map((name, index) => {
  const parts = name.split('_');
  if (parts.length === 2 && parts[0] === 'write') {
    const bitString = parts[1];
    return { type: 'write', patBits: bitString.split('').map(b => parseInt(b, 2)) as [number, number, number, number], index };
  }
  return { type: name, patBits: [0, 0, 0, 0] as [number, number, number, number], index };
});

let currentMode: 'evolve' | 'manual' | 'demo' = 'evolve';

// Manual: demo (working) canvas state and target canvas state
// let manualDemoState = new Uint8Array(GRID_SIZE * GRID_SIZE);
let manualDemoState : Uint8Array<ArrayBufferLike> = new Uint8Array(GRID_SIZE * GRID_SIZE);
let manualTargetState = new Uint8Array(GRID_SIZE * GRID_SIZE);
let manualStep = 0;
let manualAgentX = GRID_SIZE >> 1;
let manualAgentY = GRID_SIZE >> 1;

// Shared target pattern (set from Manual mode or Evolve canvas) used by Evolve and Demo
let sharedTargetPattern: Uint8Array | null = null;

// Evolve state
let isRunning = false;
let currentGeneration = 0;
let bestFitness = Infinity;
let bestSequence: Uint32Array | null = null;
let currentInitialState = new Uint8Array(GRID_SIZE * GRID_SIZE);

// Demo state
let demoPlaybackState: Uint8Array | null = null;
let demoPlaybackStep = 0;
let demoInterval: number | null = null;
let demoAgentX = GRID_SIZE >> 1;
let demoAgentY = GRID_SIZE >> 1;

// GPU device and pipeline context
let gpuDevice: GPUDevice | null = null;
let gpuQueue: GPUQueue | null = null;
let pipeline: GPUComputePipeline | null = null;
let bindGroupLayout: GPUBindGroupLayout | null = null;
let bindGroup: GPUBindGroup | null = null;

// shader string is taken from original user upload (keeps original fitness computation)
const computeShaderWGSL = `
struct Params {
  gridSize: u32,
  batchSize: u32,
  steps: u32,
  currentStep: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@group(0) @binding(3) var<storage, read> targetPattern: array<u32>;
@group(0) @binding(4) var<storage, read_write> fitness: array<u32>;

fn get_idx(x: u32, y: u32) -> u32 {
  return y * params.gridSize + x;
}

// helper to wrap coords
fn wrap_add(a: i32, b: i32, m: i32) -> i32 {
  let v = (a + b) % m;
  if (v < 0) {
    return v + m;
  }
  return v;
}

fn apply_action_move(state: ptr<function, array<u32, 144>>, action_index: u32, grid_size: u32) {
  // For move actions we store agent position in indices 140,141 as x,y (this is a convention used in input)
  // action_index: 0=up,1=down,2=left,3=right
  var ax = (*state)[140u];
  var ay = (*state)[141u];
  if (action_index == 0u) {
    ay = (ay + grid_size - 1u) % grid_size;
  } else if (action_index == 1u) {
    ay = (ay + 1u) % grid_size;
  } else if (action_index == 2u) {
    ax = (ax + grid_size - 1u) % grid_size;
  } else if (action_index == 3u) {
    ax = (ax + 1u) % grid_size;
  }
  (*state)[140u] = ax;
  (*state)[141u] = ay;
}

fn apply_action_write(state: ptr<function, array<u32, 144>>, action_index: u32, grid_size: u32) {
  // action_index 5..20 map to bit patterns for 2x2 write
  let pat = action_index - 5u; // 0..15
  var ax = (*state)[140u];
  var ay = (*state)[141u];

  for (var i: i32 = 0; i < 2; i = i + 1) {
    for (var j: i32 = 0; j < 2; j = j + 1) {
      let bit = (pat >> (u32(i*2 + j))) & 1u;
      let wx = u32(wrap_add(i32(ax) + i32(j) - 1, 0, i32(grid_size)));
      let wy = u32(wrap_add(i32(ay) + i32(i) - 1, 0, i32(grid_size)));
      (*state)[get_idx(wx, wy)] = bit;
    }
  }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let idx = global_id.x;
  let batch_idx = idx;
  if (batch_idx >= params.batchSize) {
    return;
  }

  let grid_size = params.gridSize;
  let state_size = grid_size * grid_size;

  let offset = batch_idx * state_size;
  // We'll keep agent position at reserved indices 140,141 (outside the 144 grid)
  // For convenience we allocate a local array with a margin for agent coords
  var current_state: array<u32, 144>;
  for (var i: u32 = 0; i < state_size; i = i + 1) {
    current_state[i] = input[offset + i];
  }
  // initialize agent pos (center)
  current_state[140u] = grid_size / 2u;
  current_state[141u] = grid_size / 2u;

  let sequence_offset = params.batchSize * state_size + batch_idx * params.steps;

  for (var s: u32 = 0; s < params.steps; s = s + 1) {
    let action_index = input[sequence_offset + s];

    if (action_index >= 0u && action_index <= 3u) {
        apply_action_move(&current_state, action_index, params.gridSize);
    } else if (action_index >= 5u && action_index <= 20u) {
        apply_action_write(&current_state, action_index, params.gridSize);
    } // else do_nothing or other codes - ignore

    // apply CA (Conway-like) step to current_state -> produce next_state
    var next_state: array<u32, 144>;
    for (var y: u32 = 0; y < grid_size; y = y + 1) {
      for (var x: u32 = 0; x < grid_size; x = x + 1) {
        var alive_neighbors: u32 = 0u;
        for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
          for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            if (!(dx == 0 && dy == 0)) {
              let nx_i = wrap_add(i32(x) + dx, 0, i32(grid_size));
              let ny_i = wrap_add(i32(y) + dy, 0, i32(grid_size));
              let nx = u32(nx_i);
              let ny = u32(ny_i);
              alive_neighbors = alive_neighbors + current_state[get_idx(nx, ny)];
            }
          }
        }
        let cur = current_state[get_idx(x, y)];
        var nxt: u32 = 0u;
        if (cur == 1u) {
          if (alive_neighbors == 2u || alive_neighbors == 3u) {
            nxt = 1u;
          } else {
            nxt = 0u;
          }
        } else {
          if (alive_neighbors == 3u) {
            nxt = 1u;
          } else {
            nxt = 0u;
          }
        }
        next_state[get_idx(x, y)] = nxt;
      }
    }

    // copy next_state back to current_state
    for (var i2: u32 = 0; i2 < state_size; i2 = i2 + 1) {
      current_state[i2] = next_state[i2];
    }
    // keep agent coords intact (already in 140,141)
  }

  // compute fitness: compare current_state to targetPattern
  var score: u32 = 0u;
  for (var y2: u32 = 0; y2 < grid_size; y2 = y2 + 1) {
    for (var x2: u32 = 0; x2 < grid_size; x2 = x2 + 1) {
      let idx2 = get_idx(x2, y2);
      let t = targetPattern[idx2];
      if (current_state[idx2] != t) {
        score = score + 1u;
      }
    }
  }

  fitness[batch_idx] = score;
  // write output final state for convenience (optional)
  let out_offset = batch_idx * state_size;
  for (var i3: u32 = 0; i3 < state_size; i3 = i3 + 1) {
    output[out_offset + i3] = current_state[i3];
  }
}
`;

// -----------------------
// Helper JS/TS CA functions (CPU-side fallback and helpers)
// -----------------------

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
      let next = 0;
      if (current === 1) {
        if (neighbors === 2 || neighbors === 3) next = 1;
      } else {
        if (neighbors === 3) next = 1;
      }
      nextGrid[y * size + x] = next;
    }
  }
  return nextGrid;
}

// function applyAction(state: Uint8Array, actionIndex: number, agentPos: {x: number, y: number}): Uint8Array {
function applyAction(state: Uint8Array<ArrayBufferLike>, actionIndex: number, agentPos: {x: number, y: number}): Uint8Array {
  const size = Math.sqrt(state.length);
  const newState = new Uint8Array(state);
  const action = ACTION_DECODER[actionIndex];

  if (action.type === 'do_nothing') {
    return newState;
  } else if (action.type === 'up') {
    agentPos.y = (agentPos.y - 1 + size) % size;
  } else if (action.type === 'down') {
    agentPos.y = (agentPos.y + 1) % size;
  } else if (action.type === 'left') {
    agentPos.x = (agentPos.x - 1 + size) % size;
  } else if (action.type === 'right') {
    agentPos.x = (agentPos.x + 1) % size;
  } else if (action.type === 'write') {
    const patternBits = action.patBits;
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        // Note: agent is center; write at (agentY + i - 1, agentX + j - 1)
        const yy = (agentPos.y + i - 1 + size) % size;
        const xx = (agentPos.x + j - 1 + size) % size;
        const idx = yy * size + xx;
        newState[idx] = patternBits[i * 2 + j];
      }
    }
  }
  return newState;
}

/* Simulate a sequence starting from currentInitialState and return final grid */
function evaluateSequenceToGrid(seq: Uint32Array | null): Uint8Array | null {
  if (!seq || !currentInitialState) return null;
  const size = Math.sqrt(currentInitialState.length);
//   let state = new Uint8Array(currentInitialState);
  let state : Uint8Array<ArrayBufferLike> = new Uint8Array(currentInitialState);
  let agent = { x: size >> 1, y: size >> 1 };
  for (let a = 0; a < seq.length; a++) {
    const actionIndex = seq[a];
    // apply action then conway step
    state = applyAction(state, actionIndex, agent) as Uint8Array<ArrayBufferLike>;
    state = conwayStep(state) as Uint8Array<ArrayBufferLike>;
  }
  return state;
}

/* -----------------------
   Rendering helpers
   ----------------------- */

function renderGrid(canvas: HTMLCanvasElement, grid: Uint8Array | null, cellColor: string = '#2196F3', showAgent: boolean = false, agentX: number = 0, agentY: number = 0) {
  const size = Math.sqrt(grid ? grid.length : GRID_SIZE * GRID_SIZE);
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const cell = canvas.width / size;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = '#eee';
  ctx.fillStyle = cellColor;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      if (grid && grid[y * size + x] === 1) {
        // draw with 1-pixel inset to avoid overlap artifacts
        ctx.fillRect(Math.floor(x * cell), Math.floor(y * cell), Math.ceil(cell), Math.ceil(cell));
      }
      ctx.strokeRect(Math.floor(x * cell), Math.floor(y * cell), Math.ceil(cell), Math.ceil(cell));
    }
  }

  // Draw agent position (2x2 square) centered similar to Python (agentX - 1.5, agentY - 1.5)
  if (showAgent) {
    ctx.strokeStyle = '#FF00FF';
    ctx.lineWidth = 3;
    const px = (agentX - 1.5) * cell;
    const py = (agentY - 1.5) * cell;
    ctx.strokeRect(px, py, 2 * cell, 2 * cell);
    ctx.lineWidth = 1;
  }
}

/* -----------------------
   UI wiring
   ----------------------- */

function getUIElements(): UIElements {
  const seqStripEl = document.getElementById('seqStrip') as HTMLCanvasElement;
  const dummyModeSelect = document.createElement('select') as HTMLSelectElement;
  
  ['evolve', 'manual', 'demo'].forEach(mode => {
    const option = document.createElement('option');
    option.value = mode;
    option.textContent = mode.charAt(0).toUpperCase() + mode.slice(1) + ' Mode';
    dummyModeSelect.appendChild(option);
  });
  dummyModeSelect.value = 'evolve';

  return {
    mode: dummyModeSelect,
    population: document.getElementById('population') as HTMLInputElement,
    steps: document.getElementById('steps') as HTMLInputElement,
    generations: document.getElementById('generations') as HTMLInputElement,
    elite: document.getElementById('elite') as HTMLInputElement,
    mut: document.getElementById('mut') as HTMLInputElement,
    vizFreq: document.getElementById('vizFreq') as HTMLInputElement,
    startBtn: document.getElementById('startBtn') as HTMLButtonElement,
    stopBtn: document.getElementById('stopBtn') as HTMLButtonElement,
    canvas: document.getElementById('targetVizCanvas') as HTMLCanvasElement,
    fitnessCanvas: seqStripEl,
    seqStrip: seqStripEl,
    statsDiv: document.getElementById('statsDiv') as HTMLElement,
    manualMode: document.getElementById('manual-mode') as HTMLElement,
    manualDemoCanvas: document.getElementById('manualDemoCanvas') as HTMLCanvasElement,
    manualTargetCanvas: document.getElementById('manualTargetCanvas') as HTMLCanvasElement,
    manualStats: document.getElementById('manualStats') as HTMLElement,
    vizToggle: document.getElementById('vizToggle') as HTMLInputElement,
    patternCanvas: document.getElementById('patternCanvas') as HTMLCanvasElement,
    clearPatternBtn: document.getElementById('clearPatternBtn') as HTMLButtonElement,
    savePatternBtn: document.getElementById('savePatternBtn') as HTMLButtonElement,
    demoMode: document.getElementById('demo-mode') as HTMLElement,
    demoTargetCanvas: document.getElementById('demoTargetCanvas') as HTMLCanvasElement,
    demoBestCanvas: document.getElementById('demoBestCanvas') as HTMLCanvasElement,
    demoActionsCanvas: document.getElementById('demoActionsCanvas') as HTMLCanvasElement,
    demoStep: document.getElementById('demoStep') as HTMLElement,
    demoMaxSteps: document.getElementById('demoMaxSteps') as HTMLElement,
    demoPlayBtn: document.getElementById('demoPlayBtn') as HTMLButtonElement,
    demoPauseBtn: document.getElementById('demoPauseBtn') as HTMLButtonElement,
    demoResetBtn: document.getElementById('demoResetBtn') as HTMLButtonElement,
    demoStepBtn: document.getElementById('demoStepBtn') as HTMLButtonElement,
    demoInfo: document.getElementById('demoInfo') as HTMLElement,
    log: document.getElementById('log') as HTMLElement,
    evolveBestCanvas: document.getElementById('evolveBestCanvas') as HTMLCanvasElement,
    evolveCurrentCanvas: document.getElementById('evolveCurrentCanvas') as HTMLCanvasElement,
    targetVizCanvas: document.getElementById('targetVizCanvas') as HTMLCanvasElement,
    evolveMode: document.getElementById('evolve-mode') as HTMLElement,
  };
}

function initEvolveMode(ui: UIElements): void {};

function handleModeChange(ui: UIElements) {
  document.querySelectorAll('.mode-content').forEach(el => (el as HTMLElement).style.display = 'none');

  const selectedMode = document.getElementById(`${currentMode}-mode`);
  if (selectedMode) {
    selectedMode.style.display = 'block';
  }

  if (currentMode === 'evolve') {
    initEvolveMode(ui);
  } else if (currentMode === 'manual') {
    initManualMode(ui);
  } else if (currentMode === 'demo') {
    initializeDemoMode();
  }
}

/* -----------------------
   Manual Mode
   ----------------------- */

function calculateGridStats(grid: Uint8Array): { density: number } {
  const aliveCells = grid.reduce((a, b) => a + b, 0);
  const density = aliveCells / grid.length;
  return { density };
}

function updateManualDisplay() {
  const ui = getUIElements();
  // Demo canvas: show agent and demo state
  renderGrid(ui.manualDemoCanvas, manualDemoState, '#2196F3', true, manualAgentX, manualAgentY);
  // Target canvas: editable (no agent)
  renderGrid(ui.manualTargetCanvas, manualTargetState, '#F44336', false);

  const stats = calculateGridStats(manualDemoState);
  ui.manualStats.innerHTML = `
    Step: <strong>${manualStep}/${MANUAL_MAX_STEPS}</strong><br>
    Agent Position: <strong>(${manualAgentX}, ${manualAgentY})</strong><br>
    Density: <strong>${stats.density.toFixed(2)}</strong>
  `;
}

function initManualMode(ui: UIElements) {
  // ensure canvases sized
  ui.manualDemoCanvas.width = 240;
  ui.manualDemoCanvas.height = 240;
  ui.manualTargetCanvas.width = 240;
  ui.manualTargetCanvas.height = 240;

  // clicking target canvas toggles target cells
  const cell = ui.manualTargetCanvas.width / GRID_SIZE;
  ui.manualTargetCanvas.addEventListener('click', (e) => {
    const rect = ui.manualTargetCanvas.getBoundingClientRect();
    const x = Math.floor((e.clientX - rect.left) / cell);
    const y = Math.floor((e.clientY - rect.top) / cell);
    if (x >= 0 && x < GRID_SIZE && y >= 0 && y < GRID_SIZE) {
      const idx = y * GRID_SIZE + x;
      manualTargetState[idx] = 1 - manualTargetState[idx];
      sharedTargetPattern = new Uint8Array(manualTargetState);
      updateManualDisplay();
    }
  });

  // Save current demo state into the target canvas (also update sharedTargetPattern)
  ui.savePatternBtn.addEventListener('click', () => {
    manualTargetState = new Uint8Array(manualDemoState);
    sharedTargetPattern = new Uint8Array(manualTargetState);
    log(ui.log, 'Pattern saved from demo canvas to target (shared).');
    updateManualDisplay();
  });

  // Clear target
  ui.clearPatternBtn.addEventListener('click', () => {
    manualTargetState.fill(0);
    sharedTargetPattern = new Uint8Array(manualTargetState);
    updateManualDisplay();
  });

  // keyboard controls operate on the demo canvas (agent)
  document.addEventListener('keydown', (e) => {
    if (currentMode !== 'manual') return;

    const ui = getUIElements();
    const key = e.key.toLowerCase();

    if (key === 'c') {
      // clear demo
      manualDemoState.fill(0);
      manualStep = 0;
      manualAgentX = GRID_SIZE >> 1;
      manualAgentY = GRID_SIZE >> 1;
      updateManualDisplay();
    } else if (key === 'arrowup' || key === 'arrowdown' || key === 'arrowleft' || key === 'arrowright') {
      if (manualStep >= MANUAL_MAX_STEPS) return;
      e.preventDefault();
      if (key === 'arrowup') manualAgentY = (manualAgentY - 1 + GRID_SIZE) % GRID_SIZE;
      else if (key === 'arrowdown') manualAgentY = (manualAgentY + 1) % GRID_SIZE;
      else if (key === 'arrowleft') manualAgentX = (manualAgentX - 1 + GRID_SIZE) % GRID_SIZE;
      else if (key === 'arrowright') manualAgentX = (manualAgentX + 1 + GRID_SIZE) % GRID_SIZE;

      // after moving, run one CA step
    //   manualDemoState = new Uint8Array(conwayStep(manualDemoState).buffer);
      manualDemoState = conwayStep(manualDemoState) as Uint8Array<ArrayBufferLike>;
      manualStep = Math.min(manualStep + 1, MANUAL_MAX_STEPS);
      updateManualDisplay();
    } else if (key === ' ') {
      e.preventDefault();
      if (manualStep >= MANUAL_MAX_STEPS) return;
      manualDemoState = conwayStep(manualDemoState) as Uint8Array<ArrayBufferLike>;
      manualStep = Math.min(manualStep + 1, MANUAL_MAX_STEPS);
      updateManualDisplay();
    } else if (/^[0-9a-f]$/.test(key)) {
      if (manualStep >= MANUAL_MAX_STEPS) return;
      const patternNum = parseInt(key, 16);
      const bits = [(patternNum >> 3) & 1, (patternNum >> 2) & 1, (patternNum >> 1) & 1, patternNum & 1];
      // write 2x2 centered at agent (same indexing as applyAction)
      for (let i = 0; i < 2; i++) {
        for (let j = 0; j < 2; j++) {
          const yy = (manualAgentY + i - 1 + GRID_SIZE) % GRID_SIZE;
          const xx = (manualAgentX + j - 1 + GRID_SIZE) % GRID_SIZE;
          const idx = yy * GRID_SIZE + xx;
          manualDemoState[idx] = bits[i * 2 + j];
        }
      }
      manualStep = Math.min(manualStep + 1, MANUAL_MAX_STEPS);
      updateManualDisplay();
    } else if (key === 's') {
      // shortcut key: save demo as target
      manualTargetState = new Uint8Array(manualDemoState);
      sharedTargetPattern = new Uint8Array(manualTargetState);
      log(ui.log, 'Pattern saved from demo canvas to target (shared).');
      updateManualDisplay();
    }
  });

  // init display
  updateManualDisplay();
}

/* -----------------------
   Evolve Mode (GPU-integrated)
   ----------------------- */

function log(el: HTMLElement | null, ...args: any[]) {
  if (!el) return;
  el.textContent += args.join(' ') + '\n';
  el.scrollTop = el.scrollHeight;
}

async function requestDevice(): Promise<GPUDevice> {
  if (!navigator.gpu) throw new Error('WebGPU not supported');
  if (gpuDevice) return gpuDevice;
  const adapter = await (navigator as any).gpu.requestAdapter();
  if (!adapter) throw new Error('No GPU adapter found');
  gpuDevice = await adapter.requestDevice();
  gpuQueue = gpuDevice!.queue;
  return gpuDevice!;
}

/* Setup compute pipeline using computeShaderWGSL */
async function setupComputePipeline(batchSize: number, steps: number) {
  const device = await requestDevice();
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

async function runEvolution(ui: UIElements) {
  if (!sharedTargetPattern) {
    log(ui.log, 'ERROR: No target pattern set. Switch to Manual mode and save a pattern first.');
    ui.startBtn.disabled = false;
    ui.stopBtn.disabled = true;
    return;
  }

  // prepare initial state: same as Python — small 2x2 at the center
  const center = Math.floor(GRID_SIZE / 2);
  const initial = new Uint32Array(GRID_SIZE * GRID_SIZE);
  initial[center * GRID_SIZE + center] = 1;
  initial[center * GRID_SIZE + center + 1] = 1;
  initial[(center + 1) * GRID_SIZE + center] = 1;
  initial[(center + 1) * GRID_SIZE + center + 1] = 1;
  // store as currentInitialState (Uint8 for CPU use)
  currentInitialState = new Uint8Array(initial.length);
  for (let i = 0; i < initial.length; i++) currentInitialState[i] = initial[i];

  const device = await requestDevice();
  const batchSize = Number(ui.population.value) || 50;
  const steps = Number(ui.steps.value) || 12;
  const generations = Number(ui.generations.value) || 100;
  const mutationRate = Number(ui.mut.value) || 0.1;
  const eliteFrac = Number(ui.elite.value) || 0.2;
  const vizFreq = Number(ui.vizFreq.value) || 5;

  // ensure pipeline
  await setupComputePipeline(batchSize, steps);

  // Buffer sizes: input contains states (batch * stateSize) + sequences (batch * steps)
  const stateSize = GRID_SIZE * GRID_SIZE;
  const inputStatesSize = batchSize * stateSize;
  const inputSequenceSize = batchSize * steps;
  const inputTotalSize = (inputStatesSize + inputSequenceSize);
  const inputBufferSizeBytes = inputTotalSize * 4;

  // Create buffers
  // inputBuffer (u32) — initial states + sequences; we'll fill it each generation on CPU
  const inputBuffer = device.createBuffer({
    size: inputBufferSizeBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  // outputBuffer (u32) — final states (batch * stateSize)
  const outputBuffer = device.createBuffer({
    size: inputStatesSize * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  // targetPattern buffer (u32)
  const targetBuffer = device.createBuffer({
    size: stateSize * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    mappedAtCreation: false,
  });
  // write target pattern
  const paddedTarget = new Uint32Array(stateSize);
  for (let i = 0; i < stateSize; i++) paddedTarget[i] = (sharedTargetPattern ? sharedTargetPattern[i] : 0);
  device.queue.writeBuffer(targetBuffer, 0, paddedTarget.buffer, paddedTarget.byteOffset, paddedTarget.byteLength);

  // fitnessBuffer (u32) - GPU writes fitness per individual
  const fitnessBuffer = device.createBuffer({
    size: batchSize * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  // readback buffers
  const fitnessReadBuffer = device.createBuffer({
    size: batchSize * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const outputReadBuffer = device.createBuffer({
    size: inputStatesSize * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  // params uniform buffer
  const paramsBufferSize = 4 * 4; // gridSize, batchSize, steps, currentStep
  const paramsBuffer = device.createBuffer({
    size: paramsBufferSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // pipeline and bind group
  const computePipeline = pipeline!;
  // create bind group for the pipeline — layout will depend on auto layout
  bindGroup = device.createBindGroup({
    layout: computePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: paramsBuffer } },
      { binding: 1, resource: { buffer: inputBuffer } },
      { binding: 2, resource: { buffer: outputBuffer } },
      { binding: 3, resource: { buffer: targetBuffer } },
      { binding: 4, resource: { buffer: fitnessBuffer } },
    ],
  });

  // Helper to create a population: sequences are integers 0..20 (actions) (we'll store as u32)
  function randomSequenceArray(pop: number, stepsCount: number) : Uint32Array {
    const arr = new Uint32Array(pop * stepsCount);
    for (let i = 0; i < arr.length; i++) {
      arr[i] = Math.floor(Math.random() * ACTIONS.length);
    }
    return arr;
  }

  // create initial population sequences
  let populationSequences = randomSequenceArray(batchSize, steps);
  // and initial states: we'll fill every individual's initial state with the same seed initial pattern
  let inputStates = new Uint32Array(inputStatesSize);
  for (let b = 0; b < batchSize; b++) {
    const base = b * stateSize;
    for (let i = 0; i < stateSize; i++) {
      inputStates[base + i] = initial[i];
    }
  }

  // create a typed array that represents the full inputBuffer contents (states + sequences)
  const inputFullArray = new Uint32Array(inputTotalSize);
  inputFullArray.set(inputStates, 0);
  inputFullArray.set(populationSequences, inputStatesSize);

  // Write it for the first time
  device.queue.writeBuffer(inputBuffer, 0, inputFullArray.buffer, inputFullArray.byteOffset, inputFullArray.byteLength);

  // Update uniform params
  const paramsArray = new Uint32Array([GRID_SIZE, batchSize, steps, 0]);
  device.queue.writeBuffer(paramsBuffer, 0, paramsArray.buffer, paramsArray.byteOffset, paramsArray.byteLength);

  // Prepare UI stats
  ui.statsDiv.innerHTML = `Gen: 0<br>Best Fitness: N/A<br>Avg Fitness: N/A<br>Elite: ${eliteFrac}<br>Mut rate: ${mutationRate}`;

  isRunning = true;
  currentGeneration = 0;
  bestFitness = Infinity;
  bestSequence = null;

  // For logging diversity we keep track of a small sample of sequences as strings
  const seqSet = new Set<string>();

  // Main GA loop
  for (let gen = 0; gen < generations && isRunning; gen++) {
    currentGeneration = gen;

    // write params.currentStep = 0 for full run (shader expects params.steps for inner loop)
    paramsArray[3] = 0;
    device.queue.writeBuffer(paramsBuffer, 0, paramsArray.buffer, paramsArray.byteOffset, paramsArray.byteLength);

    // refresh input buffer with current sequences and states
    inputFullArray.set(inputStates, 0);
    inputFullArray.set(populationSequences, inputStatesSize);
    device.queue.writeBuffer(inputBuffer, 0, inputFullArray.buffer, inputFullArray.byteOffset, inputFullArray.byteLength);

    // dispatch compute
    const commandEncoder = device.createCommandEncoder();
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(computePipeline);
    pass.setBindGroup(0, bindGroup!);
    const workgroupCount = Math.ceil(batchSize / 64);
    pass.dispatchWorkgroups(workgroupCount);
    pass.end();
    // copy fitness to read buffer & output to read buffer
    commandEncoder.copyBufferToBuffer(fitnessBuffer, 0, fitnessReadBuffer, 0, batchSize * 4);
    commandEncoder.copyBufferToBuffer(outputBuffer, 0, outputReadBuffer, 0, inputStatesSize * 4);
    device.queue.submit([commandEncoder.finish()]);

    // map fitnessReadBuffer and outputReadBuffer
    await fitnessReadBuffer.mapAsync(GPUMapMode.READ);
    const fitnessArrayBuffer = fitnessReadBuffer.getMappedRange();
    const fitnessArray = new Uint32Array(fitnessArrayBuffer.slice(0));
    fitnessReadBuffer.unmap();

    await outputReadBuffer.mapAsync(GPUMapMode.READ);
    const outputArrayBuffer = outputReadBuffer.getMappedRange();
    const outputArray = new Uint32Array(outputArrayBuffer.slice(0));
    outputReadBuffer.unmap();

    // compute best & avg fitness on CPU
    let minFitness = Number.MAX_SAFE_INTEGER;
    let minIdx = -1;
    let sumFitness = 0;
    for (let i = 0; i < batchSize; i++) {
      const f = fitnessArray[i];
      sumFitness += f;
      if (f < minFitness) {
        minFitness = f;
        minIdx = i;
      }
    }
    const avgFitness = sumFitness / batchSize;

    // update best if improved
    if (minFitness < bestFitness) {
      bestFitness = minFitness;
      // read sequence for minIdx from populationSequences
      const seq = new Uint32Array(steps);
      for (let s = 0; s < steps; s++) {
        seq[s] = populationSequences[minIdx * steps + s];
      }
      bestSequence = seq;
      log(ui.log, `Gen ${gen}: New best fitness ${bestFitness} (idx ${minIdx})`);
      // For demo playback, compute demoPlaybackState from bestSequence and initial
      const bestGrid = evaluateSequenceToGrid(bestSequence);
      if (bestGrid && ui.evolveBestCanvas) {
        renderGrid(ui.evolveBestCanvas, bestGrid, '#4CAF50', false);
      }
    }

    // diversity: sample small subset of sequences, add stringified
    seqSet.clear();
    const sampleCount = Math.min(20, batchSize);
    for (let s = 0; s < sampleCount; s++) {
      const idx = Math.floor(Math.random() * batchSize);
      const seqStr = Array.from(populationSequences.slice(idx * steps, (idx + 1) * steps)).join(',');
      seqSet.add(seqStr);
    }
    const diversity = seqSet.size / sampleCount;

    // update UI stats and render sample/current and target
    ui.statsDiv.innerHTML = `
      Gen: <strong>${gen}</strong><br>
      Best Fitness: <strong>${bestFitness === Infinity ? 'N/A' : bestFitness}</strong><br>
      Avg Fitness: <strong>${avgFitness.toFixed(2)}</strong><br>
      Population: <strong>${batchSize}</strong><br>
      Steps: <strong>${steps}</strong><br>
      Diversity (sample): <strong>${diversity.toFixed(2)}</strong><br>
      Mutation rate: <strong>${mutationRate}</strong>
    `;

    // always render target
    if (ui.targetVizCanvas) renderGrid(ui.targetVizCanvas, sharedTargetPattern ? new Uint8Array(sharedTargetPattern) : new Uint8Array(stateSize), '#F44336');

    // render a sample current output state (take outputArray of first individual)
    // outputArray is a big array containing batch*stateSize outputs
    if (ui.evolveCurrentCanvas) {
      const sampleOut = new Uint8Array(stateSize);
      for (let i = 0; i < stateSize; i++) sampleOut[i] = outputArray[i];
      renderGrid(ui.evolveCurrentCanvas, sampleOut, '#2196F3', false);
    }

    // early-exit if we reached perfect fitness (0 mismatches)
    if (bestFitness === 0) {
      log(ui.log, `Perfect match found in generation ${gen}.`);
      break;
    }

    // GA reproduction step (simple elitism + mutation)
    // collect pairs sorted by fitness
    const indices = new Array(batchSize).fill(0).map((_, i) => i);
    indices.sort((a, b) => fitnessArray[a] - fitnessArray[b]);

    // keep elites
    const eliteCount = Math.max(1, Math.floor(batchSize * eliteFrac));
    const newPopulation = new Uint32Array(batchSize * steps);
    // copy elites directly
    for (let e = 0; e < eliteCount; e++) {
      const srcIdx = indices[e];
      newPopulation.set(populationSequences.slice(srcIdx * steps, srcIdx * steps + steps), e * steps);
    }
    // fill remaining with crossover + mutation
    for (let i = eliteCount; i < batchSize; i++) {
      // tournament selection 2
      const a = indices[Math.floor(Math.random() * Math.max(1, batchSize * 0.5))];
      const b = indices[Math.floor(Math.random() * Math.max(1, batchSize * 0.5))];
      const parentA = populationSequences.slice(a * steps, a * steps + steps);
      const parentB = populationSequences.slice(b * steps, b * steps + steps);
      const child = new Uint32Array(steps);
      const crossPoint = Math.floor(Math.random() * steps);
      for (let s = 0; s < steps; s++) {
        child[s] = (s < crossPoint) ? parentA[s] : parentB[s];
        // mutation
        if (Math.random() < mutationRate) {
          child[s] = Math.floor(Math.random() * ACTIONS.length);
        }
      }
      newPopulation.set(child, i * steps);
    }

    populationSequences = newPopulation;

    // small pause for viz responsiveness if requested
    if (gen % vizFreq === 0) {
      await new Promise(r => setTimeout(r, 1));
    }
  } // end generations

  // cleanup GPU temporary buffers (let GC handle but destroy the obvious)
  // Note: GPUBuffer.destroy() not yet supported everywhere; skip explicit destroy.

  // Update UI final
  if (bestSequence) {
    const bestGrid = evaluateSequenceToGrid(bestSequence);
    if (bestGrid && ui.evolveBestCanvas) renderGrid(ui.evolveBestCanvas, bestGrid, '#4CAF50', false);
    // store for demo
    // (bestSequence kept in global)
  }

  isRunning = false;
  ui.startBtn.disabled = false;
  ui.stopBtn.disabled = true;

  log(ui.log, 'Evolution complete. Best sequence available for Demo mode.');
}

/* -----------------------
   Demo Mode
   ----------------------- */

function updateDemoDisplay(ui: UIElements) {
  // target
  renderGrid(ui.demoTargetCanvas, sharedTargetPattern ? new Uint8Array(sharedTargetPattern) : new Uint8Array(GRID_SIZE * GRID_SIZE), '#F44336');

  if (demoPlaybackState && bestSequence) {
    renderGrid(ui.demoBestCanvas, demoPlaybackState, '#2196F3', true, demoAgentX, demoAgentY);
    ui.demoStep.textContent = demoPlaybackStep.toString();

    // Render action sequence strip
    const seqCanvas = ui.demoActionsCanvas;
    const ctx = seqCanvas.getContext('2d');
    if (!ctx) return;

    const steps = bestSequence.length;
    const itemW = seqCanvas.width / steps;

    ctx.clearRect(0, 0, seqCanvas.width, seqCanvas.height);
    for (let i = 0; i < steps; i++) {
      const a = bestSequence[i];
      if (i < demoPlaybackStep) ctx.fillStyle = '#ddd';
      else if (i === demoPlaybackStep) ctx.fillStyle = 'red';
      else ctx.fillStyle = '#f5f5f5';
      ctx.fillRect(i * itemW, 0, Math.ceil(itemW), seqCanvas.height);
      ctx.strokeRect(i * itemW, 0, Math.ceil(itemW), seqCanvas.height);
    }
  } else {
    renderGrid(ui.demoBestCanvas, demoPlaybackState || new Uint8Array(GRID_SIZE * GRID_SIZE), '#2196F3', true, demoAgentX, demoAgentY);
  }
}

function demoStepForward(ui: UIElements) {
  if (!bestSequence) return;
  if (!demoPlaybackState) demoPlaybackState = new Uint8Array(currentInitialState || new Uint8Array(GRID_SIZE * GRID_SIZE));
  if (demoPlaybackStep >= bestSequence.length) return;

  const actionIndex = bestSequence[demoPlaybackStep];
  const agentPos = { x: demoAgentX, y: demoAgentY };
  demoPlaybackState = applyAction(demoPlaybackState, actionIndex, agentPos);
  demoAgentX = agentPos.x;
  demoAgentY = agentPos.y;

  demoPlaybackState = conwayStep(demoPlaybackState);

  demoPlaybackStep++;
  updateDemoDisplay(ui);

  if (demoPlaybackStep >= bestSequence.length) {
    ui.demoInfo.textContent = 'Sequence finished.';
    if (demoInterval !== null) {
      clearInterval(demoInterval);
      demoInterval = null;
    }
    ui.demoPlayBtn.disabled = true;
    ui.demoPauseBtn.disabled = true;
  }
}

function resetDemo(ui: UIElements) {
  if (demoInterval !== null) {
    clearInterval(demoInterval);
    demoInterval = null;
  }
  
  demoPlaybackState = new Uint8Array(currentInitialState || new Uint8Array(GRID_SIZE * GRID_SIZE));
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
  ui.demoTargetCanvas.width = 240;
  ui.demoTargetCanvas.height = 240;
  ui.demoBestCanvas.width = 240;
  ui.demoBestCanvas.height = 240;
  ui.demoActionsCanvas.width = 400;
  ui.demoActionsCanvas.height = 40;

  if (!sharedTargetPattern || !currentInitialState || !bestSequence) {
    ui.demoInfo.textContent = 'No data available. Run Evolve Mode first to generate a sequence.';
    demoPlaybackState = new Uint8Array(GRID_SIZE * GRID_SIZE);
    ui.demoPlayBtn.disabled = true;
    ui.demoStepBtn.disabled = true;
  } else {
    ui.demoMaxSteps.textContent = bestSequence.length.toString();
    ui.demoInfo.textContent = `Loaded sequence with ${bestSequence.length} steps.`;
    ui.demoPlayBtn.disabled = false;
    ui.demoStepBtn.disabled = false;
  }

  ui.demoPlayBtn.addEventListener('click', () => {
    if (!bestSequence) return;
    if (demoPlaybackStep >= bestSequence!.length) resetDemo(ui);
    ui.demoPlayBtn.disabled = true;
    ui.demoPauseBtn.disabled = false;
    ui.demoStepBtn.disabled = true;
    demoInterval = setInterval(() => demoStepForward(ui), 500) as unknown as number;
  });

  ui.demoPauseBtn.addEventListener('click', () => {
    if (demoInterval !== null) {
      clearInterval(demoInterval);
      demoInterval = null;
    }
    ui.demoPlayBtn.disabled = false;
    ui.demoPauseBtn.disabled = true;
    ui.demoStepBtn.disabled = false;
  });

  ui.demoResetBtn.addEventListener('click', () => {
    resetDemo(ui);
    ui.demoInfo.textContent = 'Demo reset.';
  });

  ui.demoStepBtn.addEventListener('click', () => {
    if (bestSequence && demoPlaybackStep < bestSequence.length) {
      demoStepForward(ui);
    }
  });

  resetDemo(ui);
}

/* -----------------------
   App Initialization
   ----------------------- */

function initApp() {
  const ui = getUIElements();
  
  const evolveBtn = document.getElementById('btn-evolve') as HTMLButtonElement;
  const manualBtn = document.getElementById('btn-manual') as HTMLButtonElement;
  const demoBtn = document.getElementById('btn-demo') as HTMLButtonElement;

  const handleModeButtonClick = (e: MouseEvent) => {
    const target = e.currentTarget as HTMLElement;
    const newMode = target.dataset.mode as typeof currentMode;
    currentMode = newMode;
    handleModeChange(ui);
    [evolveBtn, manualBtn, demoBtn].forEach(btn => btn.classList.remove('active'));
    target.classList.add('active');
  };

  evolveBtn.addEventListener('click', handleModeButtonClick);
  manualBtn.addEventListener('click', handleModeButtonClick);
  demoBtn.addEventListener('click', handleModeButtonClick);

  // default show evolve
  currentMode = 'evolve';
  handleModeChange(ui);
  evolveBtn.classList.add('active');

  ui.startBtn.addEventListener('click', async () => {
    ui.startBtn.disabled = true;
    ui.stopBtn.disabled = false;
    ui.log.textContent = '';
    try {
      await runEvolution(ui);
    } catch (err: any) {
      log(ui.log, 'Error:', err?.message ?? String(err));
    } finally {
      ui.startBtn.disabled = false;
      ui.stopBtn.disabled = true;
    }
  });

  ui.stopBtn.addEventListener('click', () => {
    isRunning = false;
    log(ui.log, 'Evolution stopped by user.');
  });

  // clicking on evolve target canvas to edit target pattern
  ui.targetVizCanvas.addEventListener('click', (e) => {
    const rect = ui.targetVizCanvas.getBoundingClientRect();
    const cell = ui.targetVizCanvas.width / GRID_SIZE;
    const x = Math.floor((e.clientX - rect.left) / cell);
    const y = Math.floor((e.clientY - rect.top) / cell);
    if (!sharedTargetPattern) sharedTargetPattern = new Uint8Array(GRID_SIZE * GRID_SIZE);
    const idx = y * GRID_SIZE + x;
    sharedTargetPattern[idx] = 1 - sharedTargetPattern[idx];
    renderGrid(ui.targetVizCanvas, sharedTargetPattern, '#F44336');
  });

  // clicking on demo target to allow edit
  ui.demoTargetCanvas.addEventListener('click', (e) => {
    const rect = ui.demoTargetCanvas.getBoundingClientRect();
    const cell = ui.demoTargetCanvas.width / GRID_SIZE;
    const x = Math.floor((e.clientX - rect.left) / cell);
    const y = Math.floor((e.clientY - rect.top) / cell);
    if (!sharedTargetPattern) sharedTargetPattern = new Uint8Array(GRID_SIZE * GRID_SIZE);
    const idx = y * GRID_SIZE + x;
    sharedTargetPattern[idx] = 1 - sharedTargetPattern[idx];
    renderGrid(ui.demoTargetCanvas, sharedTargetPattern, '#F44336');
  });

  // initial logs
  log(ui.log, 'Ready. Draw or import a target pattern by toggling cells, or switch to Manual mode to craft one.');
}

initApp();

/* EOF */

<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Evolutionary Cellular Automata Solutions</title>
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <style>
    * { box-sizing: border-box; }
    body {
      font-family: 'Inter', system-ui, Arial;
      margin: 0;
      padding: 20px;
      background: #f8f8f8;
      color: #333;
    }
    
    h1 { margin: 0 0 16px 0; font-size: 28px; color: #1e40af; }
    
    /* Mode Selector */
    .mode-selector {
      display: flex;
      gap: 12px;
      margin-bottom: 24px;
      flex-wrap: wrap;
    }
    
    .mode-btn {
      padding: 10px 20px;
      border: 2px solid #ccc;
      background: #fff;
      cursor: pointer;
      border-radius: 8px;
      font-weight: 600;
      transition: all 0.2s;
      box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .mode-btn:hover {
      border-color: #2196F3;
      color: #2196F3;
    }
    
    .mode-btn.active {
      background: #2196F3;
      color: #fff;
      border-color: #2196F3;
      box-shadow: 0 4px 8px rgba(33, 150, 243, 0.3);
    }
    
    .mode-content { display: none; }

    /* Layout Containers */
    .evo-container, .manual-container {
      display: grid;
      grid-template-columns: minmax(280px, 1fr) 320px;
      gap: 24px;
      align-items: flex-start;
    }
    .demo-container {
      display: flex;
      gap: 24px;
      flex-wrap: wrap;
    }

    /* Canvas Styles */
    canvas {
      border: 1px solid #ddd;
      background-color: #fff;
      border-radius: 6px;
      box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }

    /* explicit fixed drawing pixels — we set width/height attributes in JS */
    .pattern-canvas {
      width: 240px;
      height: 240px;
      cursor: pointer;
    }

    #demoActionsCanvas {
      width: 400px;
      height: 40px;
    }
    
    /* Labels */
    .pattern-label {
      font-weight: 700;
      margin-bottom: 8px;
      color: #1f2937;
      border-bottom: 2px solid #e5e7eb;
      padding-bottom: 4px;
    }

    /* Stats Box */
    .stats-box {
      background: #f0f4f8;
      padding: 15px;
      border-radius: 8px;
      border: 1px solid #d1d5db;
      width: 100%;
      line-height: 1.6;
      font-size: 14px;
      box-shadow: inset 0 1px 3px rgba(0,0,0,0.05);
    }
    .stats-box strong { color: #059669; }
    
    /* Controls Styling */
    .controls {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-bottom: 16px;
    }
    
    button {
      padding: 10px 15px;
      border-radius: 6px;
      font-weight: 600;
      cursor: pointer;
      transition: background-color 0.2s;
      border: 1px solid #3b82f6;
      background-color: #3b82f6;
      color: white;
    }
    button:hover:not(:disabled) {
      background-color: #2563eb;
    }
    button:disabled {
      background-color: #9ca3af;
      border-color: #9ca3af;
      cursor: not-allowed;
    }
    
    input[type="number"], select, input[type="checkbox"] {
        padding: 6px;
        border: 1px solid #ccc;
        border-radius: 4px;
    }
    
    .param-grid {
      display: grid; 
      grid-template-columns: 1fr 1fr; 
      gap: 8px;
      margin-bottom: 12px;
    }

    .param-grid label {
      font-size: 13px;
      font-weight: 500;
    }

    .param-grid input {
      width: 100%;
    }
    
    #log {
        max-height: 200px;
        overflow-y: auto;
        background: #fff;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 6px;
        font-size: 12px;
        white-space: pre-wrap;
        font-family: monospace;
    }

    .demo-section {
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
      .evo-container, .manual-container {
        grid-template-columns: 1fr;
      }
      .demo-container {
        flex-direction: column;
      }
    }
  </style>
  <script type="module" src="./dist/app.js"></script>
</head>
<body>
  <h1>Evolving Cellular Automaton Solutions (w/ WebGPU compute shader)</h1>

  <!-- Mode Selection Buttons -->
  <div class="mode-selector">
    <button class="mode-btn" data-mode="evolve" id="btn-evolve">Evolve Mode</button>
    <button class="mode-btn" data-mode="manual" id="btn-manual">Manual Mode</button>
    <button class="mode-btn" data-mode="demo" id="btn-demo">Demo Mode</button>
  </div>

  <!-- ===== EVOLVE MODE ===== -->
  <div class="mode-content" id="evolve-mode">
    <div class="evo-container">
      <div class="evo-left">
        <div style="display:flex; gap:12px; align-items:flex-start;">
          <div style="flex:1">
            <div class="pattern-label">Target Pattern (Click to Edit)</div>
            <canvas id="targetVizCanvas" class="pattern-canvas" width="240" height="240"></canvas>
          </div>

          <div style="width:240px;">
            <div class="pattern-label">Best Solution (so far)</div>
            <canvas id="evolveBestCanvas" class="pattern-canvas" width="240" height="240"></canvas>

            <div class="pattern-label" style="margin-top:12px;">Generation Sample</div>
            <canvas id="evolveCurrentCanvas" class="pattern-canvas" width="240" height="240"></canvas>
          </div>
        </div>

        <div class="pattern-label" style="margin-top: 16px;">Evolution Log</div>
        <div id="log"></div>
      </div>

      <div class="evo-right">
        <div class="pattern-label">Controls & Parameters</div>
        <div class="controls">
          <button id="startBtn">Start Evolution</button>
          <button id="stopBtn" disabled>Stop Evolution</button>
        </div>

        <div class="stats-box" style="margin-bottom: 16px;">
          <p style="font-weight: 700; margin-bottom: 8px; margin-top: 0;">Evolution Settings:</p>
          <div class="param-grid">
            <label>Population:</label>
            <input type="number" id="population" value="50" min="10" />
            
            <label>Steps:</label>
            <input type="number" id="steps" value="12" min="1" />
            
            <label>Generations:</label>
            <input type="number" id="generations" value="100" min="1" />
            
            <label>Elite Fraction:</label>
            <input type="number" id="elite" value="0.2" min="0" max="1" step="0.1" />
            
            <label>Mutation Rate:</label>
            <input type="number" id="mut" value="0.1" min="0" max="1" step="0.01" />
            
            <label>Viz Frequency:</label>
            <input type="number" id="vizFreq" value="5" min="1" />
          </div>
          <label style="display: flex; align-items: center; gap: 8px; font-size: 13px;">
            <input type="checkbox" id="vizToggle" checked />
            Enable Visualization
          </label>
        </div>

        <div class="pattern-label">Evolution Stats</div>
        <div class="stats-box" id="statsDiv">
          Click cells above to create a target pattern, or use Manual mode.
        </div>
      </div>
    </div>
  </div>

  <!-- ===== MANUAL MODE ===== -->
  <div class="mode-content" id="manual-mode">
    <div style="display:flex; gap:24px; align-items:flex-start; margin-bottom:16px;">
      <div style="flex:1">
        <div class="pattern-label">Demo Canvas (editable via keyboard actions)</div>
        <canvas id="manualDemoCanvas" class="pattern-canvas" width="240" height="240"></canvas>
      </div>

      <div style="width:240px;">
        <div class="pattern-label">Target Pattern (Click to Edit)</div>
        <canvas id="manualTargetCanvas" class="pattern-canvas" width="240" height="240"></canvas>

        <div style="margin-top:12px; display:flex; gap:8px;">
          <button id="savePatternBtn">Save Demo → Target (S)</button>
          <button id="clearPatternBtn">Clear Target (C)</button>
        </div>
      </div>
    </div>

    <div class="manual-container">
      <div class="manual-left">
        <div class="pattern-label">Instructions</div>
        <div class="stats-box">
          <strong>Controls (Manual Mode):</strong><br>
          • Arrow Keys: Move agent on the demo canvas<br>
          • Space: Evolve CA one step (on demo canvas)<br>
          • 0-F: Write 2×2 pattern centered at the agent on the demo canvas<br>
          • S: Copy demo canvas into target pattern (the target canvas is editable by clicking)<br>
          • C: Clear target canvas
        </div>
      </div>

      <div class="manual-right">
        <div class="pattern-label">Current Stats</div>
        <div class="stats-box" id="manualStats">
          Step: <strong>0/10</strong><br>
          Agent Position: <strong>(6, 6)</strong><br>
          Density: <strong>0.00</strong>
        </div>
      </div>
    </div>
  </div>

  <!-- ===== DEMO MODE ===== -->
  <div class="mode-content" id="demo-mode">
    <div class="demo-container">
      <div class="demo-section">
        <div class="pattern-label">Target Pattern</div>
        <canvas id="demoTargetCanvas" class="pattern-canvas" width="240" height="240"></canvas>
      </div>

      <div class="demo-section">
        <div class="pattern-label">Playback (Step: <span id="demoStep">0</span>/<span id="demoMaxSteps">0</span>)</div>
        <canvas id="demoBestCanvas" class="pattern-canvas" width="240" height="240"></canvas>
        
        <div class="controls">
          <button id="demoPlayBtn">Play</button>
          <button id="demoPauseBtn" disabled>Pause</button>
          <button id="demoStepBtn">Step</button>
          <button id="demoResetBtn">Reset</button>
        </div>
      </div>

      <div class="demo-section" style="min-width: 400px;">
        <div class="pattern-label">Action Sequence</div>
        <canvas id="demoActionsCanvas" width="400" height="40"></canvas>
        
        <div class="pattern-label" style="margin-top: 16px;">Info</div>
        <div class="stats-box" id="demoInfo">
          No sequence loaded. Run Evolve Mode first.
        </div>
      </div>
    </div>
  </div>

  <!-- Hidden elements for compatibility -->
  <canvas id="seqStrip" style="display: none;"></canvas>
  <canvas id="patternCanvas" style="display: none;"></canvas>
</body>
</html>

python file for reference:
#!/usr/bin/env python3
"""
evo_ca.py

Evolutionary Algorithm for Cellular Automata Pattern Matching
Uses genetic algorithms to evolve sequences of actions that create target patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
from scipy.signal import convolve2d
from tqdm import tqdm
import argparse
import os
import time
from collections import deque
import json

plt.ion()

# --- Cellular Automata Environment ---
class CAEnv:
    """Represents the Cellular Automata (CA) environment."""
    def __init__(self, grid_size=12, initial_density=0.4, rules_name='conway',
                 reward_type='pattern', target_pattern=None, max_steps=10):
        self.grid_size = grid_size
        self.initial_density = initial_density
        self.rules_name = rules_name
        self.reward_type = reward_type
        self.max_steps = max_steps
        self.current_step = 0

        self.ca_rules = {
            'conway': {'birth': [3], 'survive': [2, 3]},
            'seeds': {'birth': [2], 'survive': []},
            'maze': {'birth': [3], 'survive': [1, 2, 3, 4, 5]}
        }
        self.rules = self.ca_rules[self.rules_name]

        self.actions = ['up', 'down', 'left', 'right', 'do_nothing'] + [f'write_{i:04b}' for i in range(16)]
        self.num_actions = len(self.actions)

        self.ca_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int32)

        if target_pattern is not None:
            self.target_pattern = target_pattern
        else:
            self.target_pattern = None

        self.reset()

    def _apply_ca_rules_fast(self, grid):
        """Fast CA rule application using convolution."""
        neighbor_counts = convolve2d(grid, self.ca_kernel, mode='same', boundary='wrap')

        birth_mask = np.isin(neighbor_counts, self.rules['birth']) & (grid == 0)
        survive_mask = np.isin(neighbor_counts, self.rules['survive']) & (grid == 1)

        new_grid = np.zeros_like(grid)
        new_grid[birth_mask | survive_mask] = 1

        return new_grid

    def reset(self):
        """Resets the environment to an initial state."""
        self.ca_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.agent_x = self.grid_size // 2
        self.agent_y = self.grid_size // 2
        self.current_step = 0
        return self.ca_grid.copy()

    def step(self, action):
        """Executes one time step in the environment."""
        current_pattern = None

        # Execute Agent Action
        if 0 <= action <= 3:
            if action == 0: self.agent_y = (self.agent_y - 1 + self.grid_size) % self.grid_size
            elif action == 1: self.agent_y = (self.agent_y + 1) % self.grid_size
            elif action == 2: self.agent_x = (self.agent_x - 1 + self.grid_size) % self.grid_size
            elif action == 3: self.agent_x = (self.agent_x + 1) % self.grid_size
        elif action == 4:
            pass
        elif action >= 5:
            pattern_index = action - 5
            bits = [(pattern_index >> 3) & 1, (pattern_index >> 2) & 1, (pattern_index >> 1) & 1, pattern_index & 1]
            current_pattern = np.array(bits).reshape(2, 2)

        # Update Cellular Automata
        self._update_ca_fast(current_pattern)

        self.current_step += 1
        done = (self.current_step >= self.max_steps)

        return self.ca_grid.copy(), done

    def _update_ca_fast(self, write_pattern=None):
        """Fast CA update using convolution."""
        self.ca_grid = self._apply_ca_rules_fast(self.ca_grid)

        # Apply write pattern after CA update
        if write_pattern is not None:
            for i in range(2):
                for j in range(2):
                    y = (self.agent_y + i - 1 + self.grid_size) % self.grid_size
                    x = (self.agent_x + j - 1 + self.grid_size) % self.grid_size
                    self.ca_grid[y, x] = write_pattern[i, j]

    def calculate_fitness(self, final_grid):
        """Calculate fitness based on match with target pattern."""
        if self.target_pattern is None:
            return 0.0

        # Perfect match bonus
        match_fraction = np.mean(final_grid == self.target_pattern)
        fitness = match_fraction * 100

        # Extra bonus for perfect match
        # if match_fraction == 1.0:
        #     fitness += 100

        return fitness

    def load_pattern(self, filename):
        """Load pattern from file."""
        if os.path.exists(filename):
            self.target_pattern = np.load(filename)
            print(f"Pattern loaded from {filename}")
            return True
        return False

    def save_pattern(self, filename):
        """Save current grid as target pattern."""
        if filename:
            np.save(filename, self.ca_grid)
            print(f"Pattern saved to {filename}")


# --- Evolutionary Algorithm ---
class EvolutionaryOptimizer:
    """Evolutionary algorithm to optimize action sequences."""

    def __init__(self, env, steps=10, population_size=100,
                 elite_fraction=0.2, mutation_rate=0.1):
        self.env = env
        self.steps = steps
        self.population_size = population_size
        self.elite_size = int(population_size * elite_fraction)
        self.mutation_rate = mutation_rate
        self.num_actions = env.num_actions

        # Initialize random population
        self.population = [self._random_sequence() for _ in range(population_size)]
        self.fitness_scores = np.zeros(population_size)

        # Track best solutions
        self.best_sequence = None
        self.best_fitness = -float('inf')
        self.best_history = []
        self.overall_best = []  # list of (sequence, fitness, generation_index)

        # Statistics
        self.generation = 0
        self.avg_fitness_history = []
        self.max_fitness_history = []
        self.diversity_history = []
        self.unique_sequences_seen = set()


    def _random_sequence(self):
        """Generate a random action sequence."""
        return np.random.randint(0, self.num_actions, size=self.steps)

    def evaluate_sequence(self, sequence):
        """Evaluate fitness of an action sequence."""
        self.env.reset()

        for action in sequence:
            _, done = self.env.step(action)
            if done:
                break

        final_grid = self.env.ca_grid.copy()
        fitness = self.env.calculate_fitness(final_grid)

        return fitness, final_grid

    # def evaluate_population(self):
    #     """Evaluate all sequences in the population."""
    #     for i in range(self.population_size):
    #         self.fitness_scores[i], _ = self.evaluate_sequence(self.population[i])

    #         # Track best solution           
    #         if self.fitness_scores[i] > self.best_fitness:
    #             self.best_fitness = self.fitness_scores[i]
    #             self.best_sequence = self.population[i].copy()
    #             # Track in overall leaderboard
    #             self.overall_best.append((self.best_sequence.copy(), self.best_fitness))
    #             self.overall_best = sorted(self.overall_best, key=lambda x: x[1], reverse=True)[:5]

    #     # Update statistics
    #     self.avg_fitness_history.append(np.mean(self.fitness_scores))
    #     self.max_fitness_history.append(np.max(self.fitness_scores))
    #     self.diversity_history.append(self._calculate_diversity())

    # def evaluate_population(self):
    #     """Evaluate all sequences in the population."""
    #     for i in range(self.population_size):
    #         self.fitness_scores[i], _ = self.evaluate_sequence(self.population[i])

    #         # Track current generation best
    #         if self.fitness_scores[i] > self.best_fitness:
    #             self.best_fitness = self.fitness_scores[i]
    #             self.best_sequence = self.population[i].copy()

    #     # Update statistics
    #     self.avg_fitness_history.append(np.mean(self.fitness_scores))
    #     self.max_fitness_history.append(np.max(self.fitness_scores))
    #     self.diversity_history.append(self._calculate_diversity())

    #     # --- Update both leaderboards ---
    #     # Get top 5 of this generation
    #     gen_top5 = self.get_top_sequences(k=5)

    #     # Merge them into the overall leaderboard (bounded at 10 entries)
    #     all_candidates = set(self.overall_best + gen_top5)
    #     self.overall_best = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:5]

    def evaluate_population(self):
        """Evaluate all sequences in the population."""
        for i in range(self.population_size):
            self.fitness_scores[i], _ = self.evaluate_sequence(self.population[i])

            # Track global uniqueness
            self.unique_sequences_seen.add(tuple(self.population[i].tolist()))

            if self.fitness_scores[i] > self.best_fitness:
                self.best_fitness = self.fitness_scores[i]
                self.best_sequence = self.population[i].copy()

        # Update statistics
        self.avg_fitness_history.append(np.mean(self.fitness_scores))
        self.max_fitness_history.append(np.max(self.fitness_scores))
        self.diversity_history.append(self._calculate_diversity())

        # --- Update both leaderboards ---
        gen_top5 = self.get_top_sequences(k=5)

        # Add generation index to top 5
        gen_top5_with_gen = [(seq, fitness, self.generation) for seq, fitness in gen_top5]

        # Combine with previous overall list
        combined = self.overall_best + gen_top5_with_gen

        # Remove duplicates by hashing sequence tuples
        unique = {}
        for seq, fitness, gen_idx in combined:
            key = tuple(seq.tolist())
            # Keep the earliest generation and highest fitness for duplicates
            if key not in unique or fitness > unique[key][1]:
                unique[key] = (seq, fitness, gen_idx)

        # Sort unique entries by fitness descending
        sorted_unique = sorted(unique.values(), key=lambda x: x[1], reverse=True)

        # Keep top 5
        self.overall_best = sorted_unique[:5]


    def _calculate_diversity(self):
        """Calculate population diversity (unique sequences)."""
        unique = len(set(tuple(seq) for seq in self.population))
        return unique / self.population_size

    def get_top_sequences(self, k=5):
        """Get top k sequences by fitness."""
        top_indices = np.argsort(self.fitness_scores)[-k:][::-1]
        return [(self.population[i].copy(), self.fitness_scores[i]) for i in top_indices]

    def select_parents(self):
        """Select elite individuals for breeding."""
        elite_indices = np.argsort(self.fitness_scores)[-self.elite_size:]
        return [self.population[i].copy() for i in elite_indices]

    def crossover(self, parent1, parent2):
        """Single-point crossover between two parents."""
        crossover_point = np.random.randint(1, self.steps)
        child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        return child

    def mutate(self, sequence):
        """Mutate a sequence by randomly changing some actions."""
        mutated = sequence.copy()
        for i in range(self.steps):
            if np.random.random() < self.mutation_rate:
                mutated[i] = np.random.randint(0, self.num_actions)
        return mutated

    def evolve(self):
        """Perform one generation of evolution."""
        # Select elite parents
        parents = self.select_parents()

        # Generate new population
        new_population = parents.copy()  # Keep elites

        # Generate offspring through crossover and mutation
        while len(new_population) < self.population_size:
            parent1, parent2 = np.random.choice(len(parents), size=2, replace=False)
            child = self.crossover(parents[parent1], parents[parent2])
            child = self.mutate(child)
            new_population.append(child)

        self.population = new_population[:self.population_size]
        self.generation += 1


# --- Visualization Helpers ---
def create_action_images(num_actions, size=20):
    """
    Pre-generates improved images for each possible action, ensuring they are not cut off.
    The 'do nothing' action is represented by an empty set symbol (∅).
    """
    action_images = {}

    # --- Arrows (0-3) ---
    # Create arrow inside a slightly smaller canvas and pad it to prevent cutoff
    inner_size = size - 4 # Use a larger margin for clarity
    img = np.zeros((inner_size, inner_size))
    mid = inner_size // 2
    
    # Up Arrow (0)
    head_size = inner_size // 3
    # Draw arrowhead
    for i in range(head_size):
        img[i, mid - i : mid + i + 1] = 1
    # Draw shaft
    img[head_size:, mid - 1 : mid + 2] = 1
    # Pad the smaller canvas to the final size
    action_images[0] = np.pad(img, 2, 'constant', constant_values=0)

    # Down Arrow (1)
    action_images[1] = np.flipud(action_images[0])

    # Left Arrow (2)
    action_images[2] = np.rot90(action_images[0], 1)

    # Right Arrow (3)
    action_images[3] = np.rot90(action_images[0], -1)

    # --- Wait/Do Nothing (4) -> Empty Set symbol ---
    img = np.zeros((size, size))
    center = size // 2
    radius = size // 3 + 1
    yy, xx = np.ogrid[-center:size-center, -center:size-center]
    dist_sq = xx*xx + yy*yy
    
    # Draw the circle (as a stroke)
    circle_mask = (dist_sq <= radius*radius) & (dist_sq >= (radius-2)**2)
    
    # Draw the diagonal slash (top-left to bottom-right)
    y_idx, x_idx = np.indices((size, size))
    # Equation for a line y=x through the center
    line_mask = abs(y_idx - x_idx) < 2
    
    # Combine the circle and the slash
    img[circle_mask | line_mask] = 1
    action_images[4] = img

    # --- Write Patterns (5 to 20) ---
    for i in range(16):
        action_id = i + 5
        bits = [(i >> 3) & 1, (i >> 2) & 1, (i >> 1) & 1, i & 1]
        pattern = np.array(bits).reshape(2, 2)
        
        # Scale up pattern and add a 1px border for visual separation
        cell_size = (size - 2) // 2
        pattern_img = np.kron(pattern, np.ones((cell_size, cell_size)))
        pattern_img = np.pad(pattern_img, 1, 'constant', constant_values=0)
        action_images[action_id] = pattern_img

    return action_images

def format_seq(seq, action_labels, max_len=15):
    """Return abbreviated sequence string if too long."""
    if len(seq) <= max_len:
        return ' '.join([action_labels[a] for a in seq])
    half = max_len // 2
    return ' '.join([action_labels[a] for a in seq[:half]]) + " … " + ' '.join([action_labels[a] for a in seq[-half:]])

def train_evolutionary(args):
    """Main evolutionary training loop with enhanced 4-row visualization."""
    print("--- Starting Evolutionary Training ---")

    seq_count = 21**args.steps

    env = CAEnv(grid_size=args.grid_size, rules_name=args.rules,
                reward_type='pattern', max_steps=args.steps)

    # Load pattern
    if args.pattern_file and os.path.exists(args.pattern_file):
        env.load_pattern(args.pattern_file)
    else:
        print("Warning: No pattern file loaded. Using empty grid as target.")

    optimizer = EvolutionaryOptimizer(
        env=env,
        steps=args.steps,
        population_size=args.population_size,
        elite_fraction=args.elite_fraction,
        mutation_rate=args.mutation_rate
    )

    # Track all unique sequences seen globally
    optimizer.unique_sequences_seen = set()

    # Setup live plotting
    if args.live_plot is not None:
        action_images = create_action_images(optimizer.num_actions, size=20)

        # fig = plt.figure(figsize=(22, 13))
        # # gs = fig.add_gridspec(4, 6, height_ratios=[1, 1, 0.8, 0.6],
        # #                       hspace=0.6, wspace=0.45)

        # # gs = fig.add_gridspec(4, 6, height_ratios=[1, 1, 0.8, 0.65], hspace=1.65, wspace=0.45)

        # gs = fig.add_gridspec(
        #     4, 6,
        #     height_ratios=[1, 1, 0.9, 0.8],  # give 4th row plenty of height
        #     hspace=0.8,  # extra vertical space to avoid overlap
        #     wspace=0.45
        # )

        fig = plt.figure(figsize=(22, 14))  # slightly taller figure
        gs = fig.add_gridspec(
            4, 6,
            height_ratios=[1, 1, 0.9, 1.1],  # 4th row taller than others
            hspace=1.1,                      # more vertical padding between rows
            wspace=0.5
        )
        plt.subplots_adjust(top=0.92, bottom=0.05)  # adds padding from window edges

        # Row 1
        ax_best = fig.add_subplot(gs[0, 0])
        ax_target = fig.add_subplot(gs[0, 1])
        ax_current = fig.add_subplot(gs[0, 2])
        ax_fitness = fig.add_subplot(gs[0, 3:])

        # Row 2
        ax_diversity = fig.add_subplot(gs[1, 0])
        ax_actions = fig.add_subplot(gs[1, 1])
        ax_dist = fig.add_subplot(gs[1, 2])
        ax_fitness_dist = fig.add_subplot(gs[1, 3:])

        # Row 3 (Leaderboards + Stats)
        ax_leaderboard_gen = fig.add_subplot(gs[2, 0:2])
        ax_leaderboard_all = fig.add_subplot(gs[2, 2:4])
        ax_info = fig.add_subplot(gs[2, 4:6])

        # Row 4 (Full-width best sequence)
        ax_seq_imgs = fig.add_subplot(gs[3, :])

        fig.suptitle('Evolutionary Algorithm Progress', fontsize=16, fontweight='bold')

        # --- First Row Displays ---
        best_img = ax_best.imshow(np.zeros((env.grid_size, env.grid_size)),
                                  cmap='binary', vmin=0, vmax=1, interpolation='nearest')
        ax_best.set_title('Best Solution', fontweight='bold')
        ax_best.set_xticks([]); ax_best.set_yticks([])

        target_img = ax_target.imshow(env.target_pattern if env.target_pattern is not None else np.zeros((env.grid_size, env.grid_size)),
                                      cmap='binary', vmin=0, vmax=1, interpolation='nearest')
        ax_target.set_title('Target Pattern', fontweight='bold')
        ax_target.set_xticks([]); ax_target.set_yticks([])

        current_img = ax_current.imshow(np.zeros((env.grid_size, env.grid_size)),
                                        cmap='binary', vmin=0, vmax=1, interpolation='nearest')
        ax_current.set_title(f"Generation Sample (Step {args.steps})", fontweight='bold')
        ax_current.set_xticks([]); ax_current.set_yticks([])

        line_max = ax_fitness.plot([], [], 'b-', linewidth=2, label='Max')[0]
        line_avg = ax_fitness.plot([], [], 'g-', linewidth=2, label='Avg')[0]
        ax_fitness.set_title('Fitness Progress', fontweight='bold')
        ax_fitness.set_xlabel('Generation'); ax_fitness.set_ylabel('Fitness')
        ax_fitness.legend(); ax_fitness.grid(True, alpha=0.3)

        # --- Second Row Displays ---
        line_div = ax_diversity.plot([], [], 'orange', linewidth=2)[0]
        ax_diversity.set_title('Population Diversity', fontweight='bold')
        ax_diversity.set_xlabel('Generation'); ax_diversity.set_ylabel('Unique Sequences %')
        ax_diversity.set_ylim([0, 1]); ax_diversity.grid(True, alpha=0.3)

        action_labels = ['↑', '↓', '←', '→', '∅'] + [f'{i:X}' for i in range(16)]
        action_colors = plt.cm.tab20(np.linspace(0, 1, len(action_labels)))

        ax_fitness_dist.set_title('Fitness Distribution', fontweight='bold')
        ax_fitness_dist.set_xlabel('Fitness'); ax_fitness_dist.set_ylabel('Count')

        ax_dist.set_title('Action Usage', fontweight='bold')
        ax_dist.set_ylabel('Frequency')

        # --- Third Row Leaderboards + Stats ---
        def format_seq(seq, labels, max_len=15):
            if len(seq) <= max_len:
                return ' '.join([labels[a] for a in seq])
            half = max_len // 2
            return ' '.join([labels[a] for a in seq[:half]]) + " … " + ' '.join([labels[a] for a in seq[-half:]])

        leaderboard_text_gen = ax_leaderboard_gen.text(
            0.02, 0.98, '', va='top', fontsize=10, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        ax_leaderboard_gen.set_title('Top 5 (This Generation)', fontweight='bold')
        ax_leaderboard_gen.axis('off')

        leaderboard_text_all = ax_leaderboard_all.text(
            0.02, 0.98, '', va='top', fontsize=10, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        ax_leaderboard_all.set_title('Top 5 (All-Time Unique)', fontweight='bold')
        ax_leaderboard_all.axis('off')

        info_text = ax_info.text(0.02, 0.98, '', va='top', fontsize=10, family='monospace')
        ax_info.set_title('Statistics', fontweight='bold')
        ax_info.axis('off')

        # --- Fourth Row (Best Sequence Visualization) ---
        ax_seq_imgs.set_title('Best Sequence (Full Visualization)', fontweight='bold', pad=10)
        ax_seq_imgs.axis('off')

        plt.pause(0.1)

    # --- Training Loop ---
    print(f"\nTraining for {args.generations} generations...")
    print(f"Population size: {args.population_size}, Sequence length: {args.steps}")
    print(f"Elite fraction: {args.elite_fraction}, Mutation rate: {args.mutation_rate}\n")

    for generation in tqdm(range(args.generations), desc="Evolution"):
        optimizer.evaluate_population()

        # Track unique sequences globally
        for seq in optimizer.population:
            optimizer.unique_sequences_seen.add(tuple(seq.tolist()))

        if args.live_plot is not None and (generation % args.live_plot == 0 or generation == args.generations - 1):
            best_fitness, best_grid = optimizer.evaluate_sequence(optimizer.best_sequence)
            best_img.set_data(best_grid)

            random_idx = np.random.randint(0, optimizer.population_size)
            _, current_grid = optimizer.evaluate_sequence(optimizer.population[random_idx])
            current_img.set_data(current_grid)

            gens = range(len(optimizer.max_fitness_history))
            line_max.set_data(gens, optimizer.max_fitness_history)
            line_avg.set_data(gens, optimizer.avg_fitness_history)
            ax_fitness.relim(); ax_fitness.autoscale_view()

            line_div.set_data(gens, optimizer.diversity_history)
            ax_diversity.relim(); ax_diversity.autoscale_view(scalex=True, scaley=False)

            ax_actions.clear()
            colors = [action_colors[a] for a in optimizer.best_sequence]
            ax_actions.bar(range(args.steps), optimizer.best_sequence, color=colors)
            ax_actions.set_title('Best Action Sequence', fontweight='bold')
            ax_actions.set_xlabel('Step'); ax_actions.set_ylabel('Action ID')
            ax_actions.set_ylim([-1, optimizer.num_actions]); ax_actions.grid(axis='y', alpha=0.3)

            # --- Best Sequence Visualization (Row 4) ---
            ax_seq_imgs.clear()
            ax_seq_imgs.set_title('Best Sequence (Full Visualization)', fontweight='bold', pad=10)
            ax_seq_imgs.axis('off')

            best_seq = optimizer.best_sequence
            if best_seq is not None and len(best_seq) > 0:
                # Build the composite image (grid of small action icons)
                seq_img_list = [action_images[action] for action in best_seq]
                padding = np.zeros_like(seq_img_list[0][:, :2])
                padded_imgs = []
                for img in seq_img_list:
                    padded_imgs.append(img)
                    padded_imgs.append(padding)
                composite_img = np.hstack(padded_imgs[:-1])

                # Keep natural aspect ratio — no stretching or scaling
                ax_seq_imgs.imshow(
                    composite_img,
                    cmap='binary',
                    interpolation='nearest',
                    aspect='equal'  # preserves square pixels
                )

                ax_seq_imgs.set_anchor('C')  # centers content in subplot
                ax_seq_imgs.set_position(ax_seq_imgs.get_position())  # lock its allocated space

                # Add small padding around edges for visual breathing room
                ax_seq_imgs.set_xlim([-2, composite_img.shape[1] + 2])
                ax_seq_imgs.set_ylim([composite_img.shape[0] + 2, -2])
            else:
                ax_seq_imgs.text(0.5, 0.5, 'No sequence', ha='center', va='center', fontsize=12)

            # --- Fitness Distribution ---
            ax_fitness_dist.clear()
            ax_fitness_dist.hist(optimizer.fitness_scores, bins=20, color='steelblue', alpha=0.7)
            ax_fitness_dist.axvline(optimizer.best_fitness, color='red', linestyle='--', linewidth=2, label='Best')
            ax_fitness_dist.set_title('Fitness Distribution', fontweight='bold')
            ax_fitness_dist.legend()

            # --- Action Usage ---
            ax_dist.clear()
            all_actions = np.concatenate(optimizer.population)
            action_counts = np.bincount(all_actions, minlength=optimizer.num_actions)
            bars = ax_dist.bar(range(optimizer.num_actions), action_counts, color='steelblue', alpha=0.7)
            best_action_counts = np.bincount(optimizer.best_sequence, minlength=optimizer.num_actions)
            for i, bar in enumerate(bars):
                if best_action_counts[i] > 0:
                    bar.set_color('coral')
            ax_dist.set_title('Action Usage', fontweight='bold')
            ax_dist.set_ylabel('Frequency')
            ax_dist.set_xticks(range(optimizer.num_actions))
            ax_dist.set_xticklabels(action_labels, fontsize=7, rotation=45)

            # --- Leaderboards ---
            top_sequences = optimizer.get_top_sequences(k=5)
            leaderboard_str_gen = "LEADERBOARD (Gen):\n" + "=" * 30 + "\n"
            for rank, (seq, fitness) in enumerate(top_sequences, 1):
                seq_str = format_seq(seq, action_labels)
                leaderboard_str_gen += f"#{rank}: {fitness:.2f}\n    {seq_str}\n"
            leaderboard_text_gen.set_text(leaderboard_str_gen)

            leaderboard_str_all = "LEADERBOARD (All-Time):\n" + "=" * 30 + "\n"
            for rank, (seq, fitness, gen_idx) in enumerate(optimizer.overall_best, 1):
                seq_str = format_seq(seq, action_labels)
                leaderboard_str_all += f"#{rank}: {fitness:.2f} (Gen {gen_idx})\n    {seq_str}\n"
            leaderboard_text_all.set_text(leaderboard_str_all)

            # --- Stats Panel ---
            info_str = (
                f"Generation: {generation + 1}/{args.generations}\n"
                f"Best Fitness: {optimizer.best_fitness:.2f}\n"
                f"Avg Fitness: {optimizer.avg_fitness_history[-1]:.2f}\n"
                f"Diversity: {optimizer.diversity_history[-1]:.2%}\n"
                f"Perfect Matches: {np.sum(optimizer.fitness_scores >= 100)}\n"
                f"Unique Sequences Seen: {len(optimizer.unique_sequences_seen)}\n"
                f"Total Number of Sequences: {seq_count}\n"
                f"Percentage of Sequences Explored: {100*len(optimizer.unique_sequences_seen)/seq_count:.8f} %\n"
            )
            info_text.set_text(info_str)

            fig.canvas.draw()
            plt.pause(0.01)

        # Evolve
        if generation < args.generations - 1:
            optimizer.evolve()

        # Save checkpoint periodically
        if (generation + 1) % args.save_freq == 0:
            checkpoint = {
                'generation': generation + 1,
                'best_sequence': optimizer.best_sequence.tolist(),
                'best_fitness': float(optimizer.best_fitness),
                'max_fitness_history': optimizer.max_fitness_history,
                'avg_fitness_history': optimizer.avg_fitness_history,
                'diversity_history': optimizer.diversity_history
            }
            with open(f'checkpoint_gen{generation+1}.json', 'w') as f:
                json.dump(checkpoint, f, indent=2)

    # --- Save final results ---
    print(f"\n--- Training Complete ---")
    print(f"Best Fitness: {optimizer.best_fitness:.2f}")
    print(f"Best Sequence: {optimizer.best_sequence}")

    results = {
        'best_sequence': optimizer.best_sequence.tolist(),
        'best_fitness': float(optimizer.best_fitness),
        'max_fitness_history': optimizer.max_fitness_history,
        'avg_fitness_history': optimizer.avg_fitness_history,
        'diversity_history': optimizer.diversity_history,
        'action_labels': action_labels
    }

    with open('evolutionary_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    np.save('best_sequence.npy', optimizer.best_sequence)

    if args.live_plot is not None:
        plt.ioff()
        plt.savefig('evolutionary_results.png', dpi=150, bbox_inches='tight')

    print("\nResults saved to evolutionary_results.json and best_sequence.npy")


def run_demo(args):
    """Demonstrate a saved action sequence."""
    print("--- Running Sequence Demo ---")

    if not os.path.exists(args.sequence_file):
        raise FileNotFoundError(f"Sequence file not found: {args.sequence_file}")

    sequence = np.load(args.sequence_file)
    print(f"Loaded sequence of length {len(sequence)}")

    env = CAEnv(grid_size=args.grid_size, rules_name=args.rules,
                reward_type='pattern', max_steps=len(sequence))
    
    action_images = create_action_images(env.num_actions, size=20)

    if args.pattern_file and os.path.exists(args.pattern_file):
        env.load_pattern(args.pattern_file)

    # Setup visualization
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[4, 4, 1], hspace=0.4, wspace=0.3)

    ax_grid = fig.add_subplot(gs[0, 0])
    ax_target = fig.add_subplot(gs[0, 1])
    ax_actions = fig.add_subplot(gs[1, 0])
    ax_info = fig.add_subplot(gs[1, 1])
    ax_seq_imgs = fig.add_subplot(gs[2, :])

    # Target pattern
    if env.target_pattern is not None:
        ax_target.imshow(env.target_pattern, cmap='binary', vmin=0, vmax=1, interpolation='nearest')
    ax_target.set_title('Target Pattern', fontweight='bold')
    ax_target.set_xticks([])
    ax_target.set_yticks([])

    # Grid
    state = env.reset()
    grid_img = ax_grid.imshow(state, cmap='binary', vmin=0, vmax=1, interpolation='nearest')
    agent_patch = plt.Rectangle((env.agent_x - 1.5, env.agent_y - 1.5), 2, 2,
                                facecolor='none', edgecolor='cyan', linewidth=2)
    ax_grid.add_patch(agent_patch)
    title_text = ax_grid.set_title("Step 0", fontweight='bold')
    ax_grid.set_xticks([])
    ax_grid.set_yticks([])

    # Action sequence bar chart
    action_labels = ['↑', '↓', '←', '→', '∅'] + [f'{i:X}' for i in range(16)]
    action_colors = plt.cm.tab20(np.linspace(0, 1, len(action_labels)))
    colors = [action_colors[a] for a in sequence]
    bars = ax_actions.bar(range(len(sequence)), sequence, color=colors)
    ax_actions.set_title('Action Sequence', fontweight='bold')
    ax_actions.set_xlabel('Step')
    ax_actions.set_ylabel('Action ID')
    ax_actions.grid(axis='y', alpha=0.3)

    # Info
    info_text = ax_info.text(0.05, 0.95, '', va='top', fontsize=11, family='monospace')
    ax_info.set_title('Info', fontweight='bold')
    ax_info.axis('off')

    # Action image sequence
    ax_seq_imgs.set_title('Action History', fontweight='bold')
    ax_seq_imgs.axis('off')

    def update(frame):
        # Reset environment at the start of each loop
        if frame == 0:
            env.reset()
            grid_img.set_data(env.ca_grid)
            agent_patch.set_xy((env.agent_x - 1.5, env.agent_y - 1.5))
            title_text.set_text("Step 0")

            # Reset bar colors
            for i, bar in enumerate(bars):
                bar.set_color(colors[i])
                bar.set_alpha(1.0)
            
            # Clear image sequence
            ax_seq_imgs.clear()
            ax_seq_imgs.set_title('Action History', fontweight='bold')
            ax_seq_imgs.axis('off')

            info_text.set_text("Step: 0\nStarting...")

        elif frame <= len(sequence):
            action = sequence[frame - 1]
            state, _ = env.step(action)

            grid_img.set_data(state)
            agent_patch.set_xy((env.agent_x - 1.5, env.agent_y - 1.5))
            title_text.set_text(f"Step {frame}")

            # Highlight current action
            for i, bar in enumerate(bars):
                if i == frame - 1:
                    bar.set_color('red')
                    bar.set_alpha(1.0)
                elif i < frame - 1:
                    bar.set_color(colors[i])
                    bar.set_alpha(0.3)
                else:
                    bar.set_color(colors[i])
                    bar.set_alpha(1.0)

            # Update action image sequence
            current_sequence = sequence[:frame]
            if len(current_sequence) > 0:
                seq_img_list = [action_images[act] for act in current_sequence]
                padding = np.zeros_like(seq_img_list[0][:, :2])
                padded_imgs = [img for act_img in seq_img_list for img in (act_img, padding)][:-1]
                composite_img = np.hstack(padded_imgs)
                ax_seq_imgs.clear()
                ax_seq_imgs.set_title('Action History', fontweight='bold')
                ax_seq_imgs.axis('off')
                ax_seq_imgs.imshow(composite_img, cmap='binary', interpolation='nearest')


            fitness = env.calculate_fitness(state)
            match_pct = np.mean(state == env.target_pattern) if env.target_pattern is not None else 0

            info_str = (
                f"Step: {frame}/{len(sequence)}\n"
                f"Action: {action_labels[action]}\n"
                f"Fitness: {fitness:.2f}\n"
                f"Match: {match_pct:.1%}\n"
                f"Alive cells: {np.sum(state)}"
            )
            if frame == len(sequence):
                info_str += f"\n\nFinal Score: {fitness:.2f}"

            info_text.set_text(info_str)

        return [grid_img, agent_patch] + list(bars) + [info_text]

    # Add pause frames at the end before looping
    total_frames = len(sequence) + 1 + 20  # 0 (reset) + sequence steps + 20 pause frames

    ani = animation.FuncAnimation(fig, update, frames=total_frames,
                                 interval=500, repeat=True, blit=False)
    plt.show(block=True)


def interactive_pattern_creator(args):
    """Interactive pattern creation tool."""
    print("--- Interactive Pattern Creator ---")
    print("Click cells to toggle them on/off")
    print("Press 's' to save pattern")
    print("Press 'c' to clear grid")
    print("Press 'q' to quit")

    grid = np.zeros((args.grid_size, args.grid_size), dtype=np.int8)

    fig, ax = plt.subplots(figsize=(8, 8))
    img = ax.imshow(grid, cmap='binary', vmin=0, vmax=1, interpolation='nearest')
    ax.set_title('Click to toggle cells | s=save | c=clear | q=quit')
    ax.set_xticks(np.arange(-.5, args.grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-.5, args.grid_size, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', size=0)


    def on_click(event):
        if event.inaxes == ax and event.button == 1:
            x = int(np.round(event.xdata))
            y = int(np.round(event.ydata))
            if 0 <= x < args.grid_size and 0 <= y < args.grid_size:
                grid[y, x] = 1 - grid[y, x]
                img.set_data(grid)
                fig.canvas.draw_idle()

    def on_key(event):
        if event.key == 's':
            filename = f'custom_pattern_{args.grid_size}x{args.grid_size}.npy'
            np.save(filename, grid)
            print(f"\nPattern saved to {filename}")
            print(f"Density: {np.mean(grid):.3f}, Live cells: {np.sum(grid)}")
        elif event.key == 'c':
            grid.fill(0)
            img.set_data(grid)
            fig.canvas.draw_idle()
            print("\nGrid cleared")
        elif event.key == 'q':
            plt.close()

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show(block=True)


def run_manual(args):
    """Manual play mode with keyboard control and pattern saving."""
    import matplotlib as mpl
    mpl.rcParams['keymap.fullscreen'] = []

    print("--- Running Manual Play Mode ---")
    env = CAEnv(grid_size=args.grid_size, rules_name=args.rules,
                reward_type='pattern', max_steps=args.steps)

    if args.pattern_file and os.path.exists(args.pattern_file):
        env.load_pattern(args.pattern_file)

    state = env.reset()

    # Setup visualization
    if env.target_pattern is not None:
        fig = plt.figure(figsize=(16, 6))
        gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)
        ax_main = fig.add_subplot(gs[:, 0])
        ax_target = fig.add_subplot(gs[:, 1])
        ax_actions = fig.add_subplot(gs[0, 2:])
        ax_metrics = fig.add_subplot(gs[1, 2:])
        ax_target.imshow(env.target_pattern, cmap='binary', vmin=0, vmax=1, interpolation='nearest')
        ax_target.set_title('Target Pattern', fontweight='bold')
        ax_target.set_xticks([])
        ax_target.set_yticks([])
    else:
        fig = plt.figure(figsize=(16, 6))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        ax_main = fig.add_subplot(gs[:, 0])
        ax_actions = fig.add_subplot(gs[0, 1:])
        ax_metrics = fig.add_subplot(gs[1, 1:])

    grid_img = ax_main.imshow(env.ca_grid, cmap='binary', vmin=0, vmax=1, interpolation='nearest')
    agent_patch = plt.Rectangle((env.agent_x - 1.5, env.agent_y - 1.5), 2, 2,
                                facecolor='none', edgecolor='cyan', linewidth=2)
    ax_main.add_patch(agent_patch)
    title_text = ax_main.set_title("Step: 0 | Manual Mode", fontweight='bold')
    ax_main.set_xticks([])
    ax_main.set_yticks([])

    action_labels = ['↑', '↓', '←', '→', '∅'] + [f'{i:X}' for i in range(16)]
    action_history = deque(maxlen=args.steps)

    bars = ax_actions.bar(range(len(action_labels)), np.zeros(len(action_labels)),
                          color='steelblue', alpha=0.7)
    ax_actions.set_ylim([0, 1])
    ax_actions.set_xticks(range(len(action_labels)))
    ax_actions.set_xticklabels(action_labels, fontsize=8)
    ax_actions.set_ylabel('Usage')
    ax_actions.set_title('Action History', fontweight='bold')
    ax_actions.grid(axis='y', alpha=0.3)

    metrics_text = ax_metrics.text(0.05, 0.95, '', va='top', fontsize=11, family='monospace')
    ax_metrics.set_title('Statistics', fontweight='bold')
    ax_metrics.axis('off')

    step_counter = [0]
    key_map = {
        'up': 0,
        'down': 1,
        'left': 2,
        'right': 3,
        ' ': 4,
    }
    hex_keys = list('0123456789abcdef')

    def on_key(event):
        if event.key is None:
            return
        key = event.key.lower()

        if key == 'q' or key == 'escape':
            plt.close()
            return
        
        if step_counter[0] >= args.steps and key not in ['c', 'q', 'escape']:
            metrics_text.set_text(f"{metrics_text.get_text().split('Controls:')[0]}\nMax steps reached!\nPress 'C' to clear or 'Q' to quit.")
            fig.canvas.draw_idle()
            return

        if key == 's':
            filename = f'custom_pattern_{args.grid_size}x{args.grid_size}.npy'
            env.save_pattern(filename)
            print(f"Current grid state saved to {filename}")
            print(f"Density: {np.mean(env.ca_grid):.3f}, Live cells: {np.sum(env.ca_grid)}")
            return

        if key == 'c':
            env.reset()
            step_counter[0] = 0
            action_history.clear()
            grid_img.set_data(env.ca_grid)
            agent_patch.set_xy((env.agent_x - 1.5, env.agent_y - 1.5))
            title_text.set_text(f"Step: 0/{args.steps} | Manual Mode")
            
            # Reset bars
            for bar in bars:
                bar.set_height(0)
            
            metrics_text.set_text("Grid cleared. Ready for new sequence.")
            fig.canvas.draw_idle()
            return

        if key in key_map:
            action = key_map[key]
        elif key in hex_keys:
            action = 5 + hex_keys.index(key)
        else:
            return

        # Execute action
        next_state, done = env.step(action)
        action_history.append(action)
        step_counter[0] += 1

        # Update visualization
        grid_img.set_data(env.ca_grid)
        agent_patch.set_xy((env.agent_x - 1.5, env.agent_y - 1.5))

        # Update action history bars
        action_counts = np.bincount(list(action_history), minlength=len(action_labels))
        max_count = max(action_counts) if max(action_counts) > 0 else 1
        normalized_counts = action_counts / max_count

        for bar, count in zip(bars, normalized_counts):
            bar.set_height(count)

        # Calculate metrics
        alive_count = int(np.sum(env.ca_grid))
        density = float(np.mean(env.ca_grid))
        
        metrics_str = (
            f"Step: {step_counter[0]}/{args.steps}\n"
            f"Action: {env.actions[action]}\n"
            f"Alive cells: {alive_count}\n"
            f"Density: {density:.3f}\n"
        )
        
        title_str = f"Step: {step_counter[0]}/{args.steps} | Manual Mode"

        if env.target_pattern is not None:
            fitness = env.calculate_fitness(env.ca_grid)
            match_pct = np.mean(env.ca_grid == env.target_pattern)
            metrics_str += (
                f"Fitness: {fitness:.2f}\n"
                f"Match: {match_pct:.1%}\n"
            )
            title_str = f"Step: {step_counter[0]}/{args.steps} | Fitness: {fitness:.2f} | Match: {match_pct:.1%}"
            if step_counter[0] == args.steps:
                metrics_str += f"\nFinal Score: {fitness:.2f}\n"
        
        if step_counter[0] >= args.steps:
            metrics_str += "\nMax steps reached!\nPress 'C' to clear."
        else:
            metrics_str += (
                f"\nControls:\n"
                f"Arrow Keys/Space: Move/Wait\n"
                f"0-F: Write patterns\n"
                f"S: Save grid as pattern\n"
                f"C: Clear grid\n"
                f"Q: Quit"
            )

        metrics_text.set_text(metrics_str)
        title_text.set_text(title_str)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('key_press_event', on_key)

    print("\nManual Play Mode Controls:")
    print("- Arrow Keys: Move agent (↑/↓/←/→)")
    print("- Space: Do nothing")
    print("- 0-F: Write 2x2 patterns (hex notation)")
    print("- S: Save current grid state as pattern file")
    print("- C: Clear grid")
    print("- Q: Quit\n")
    
    # Initial text
    metrics_text.set_text(
        f"Step: 0/{args.steps}\n\n"
        f"Controls:\n"
        f"Arrow Keys/Space: Move/Wait\n"
        f"0-F: Write patterns\n"
        f"S: Save grid as pattern\n"
        f"C: Clear grid\n"
        f"Q: Quit"
    )


    plt.show(block=True)


# --- Main Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evolutionary Algorithm for CA Pattern Matching")
    subparsers = parser.add_subparsers(dest='mode', required=True, help='Select mode')

    # Training
    train_parser = subparsers.add_parser('train', help='Run evolutionary training')
    train_parser.add_argument('--generations', type=int, default=500, help='Number of generations')
    train_parser.add_argument('--steps', type=int, default=10, help='Length of action sequences')
    train_parser.add_argument('--population-size', type=int, default=100, help='Population size')
    train_parser.add_argument('--elite-fraction', type=float, default=0.2, help='Fraction of elite individuals')
    train_parser.add_argument('--mutation-rate', type=float, default=0.1, help='Mutation probability per action')
    train_parser.add_argument('--grid-size', type=int, default=12, help='Size of CA grid')
    train_parser.add_argument('--rules', type=str, default='conway', choices=['conway', 'seeds', 'maze'])
    train_parser.add_argument('--pattern-file', type=str, required=True, help='Target pattern file')
    train_parser.add_argument('--live-plot', type=int, nargs='?', const=1, default=None,
                             help='Live plotting frequency (e.g., 1 for every gen, 10 for every 10th gen). No value means off.')
    train_parser.add_argument('--save-freq', type=int, default=100, help='Checkpoint save frequency')

    # Demo
    demo_parser = subparsers.add_parser('demo', help='Demonstrate a sequence')
    demo_parser.add_argument('--sequence-file', type=str, default='best_sequence.npy', help='Sequence file')
    demo_parser.add_argument('--pattern-file', type=str, help='Target pattern file')
    demo_parser.add_argument('--grid-size', type=int, default=12)
    demo_parser.add_argument('--rules', type=str, default='conway', choices=['conway', 'seeds', 'maze'])

    # Manual Play
    manual_parser = subparsers.add_parser('manual', help='Manual play mode with keyboard control')
    manual_parser.add_argument('--grid-size', type=int, default=12)
    manual_parser.add_argument('--rules', type=str, default='conway', choices=['conway', 'seeds', 'maze'])
    manual_parser.add_argument('--pattern-file', type=str, default=None, help='Target pattern file (optional)')
    manual_parser.add_argument('--steps', type=int, default=10, help='Maximum steps before reset is required')

    # Pattern Creator
    pattern_parser = subparsers.add_parser('create_pattern', help='Interactive pattern creator')
    pattern_parser.add_argument('--grid-size', type=int, default=12)

    args = parser.parse_args()

    if args.mode == 'train':
        train_evolutionary(args)
    elif args.mode == 'demo':
        run_demo(args)
    elif args.mode == 'manual':
        run_manual(args)
    elif args.mode == 'create_pattern':
        interactive_pattern_creator(args)

Please update the app.ts and index.html file so that it complies with the requests above and is consistent with the python code.