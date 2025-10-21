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
  manualResetBtn: HTMLButtonElement;
  actionUsageCanvas: HTMLCanvasElement;
  fitnessDistCanvas: HTMLCanvasElement;
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
let manualDemoState: Uint8Array = new Uint8Array(GRID_SIZE * GRID_SIZE);
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

// Statistics tracking
let avgFitnessHistory: number[] = [];
let maxFitnessHistory: number[] = [];

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

fn wrap_add(a: i32, b: i32, m: i32) -> i32 {
  let v = (a + b) % m;
  if (v < 0) {
    return v + m;
  }
  return v;
}

fn apply_action_move(state: ptr<function, array<u32, 144>>, action_index: u32, grid_size: u32) {
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
  let pat = action_index - 5u;
  var ax = (*state)[140u];
  var ay = (*state)[141u];

  for (var i: i32 = 0; i < 2; i = i + 1) {
    for (var j: i32 = 0; j < 2; j = j + 1) {
      let bit = (pat >> (u32(i*2 + j))) & 1u;
      let wx = u32(wrap_add(i32(ax) + i32(j), 0, i32(grid_size)));
      let wy = u32(wrap_add(i32(ay) + i32(i), 0, i32(grid_size)));
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
  var current_state: array<u32, 144>;
  for (var i: u32 = 0; i < state_size; i = i + 1) {
    current_state[i] = input[offset + i];
  }
  current_state[140u] = grid_size / 2u;
  current_state[141u] = grid_size / 2u;

  let sequence_offset = params.batchSize * state_size + batch_idx * params.steps;

  for (var s: u32 = 0; s < params.steps; s = s + 1) {
    let action_index = input[sequence_offset + s];

    if (action_index >= 0u && action_index <= 3u) {
        apply_action_move(&current_state, action_index, params.gridSize);
    } else if (action_index >= 5u && action_index <= 20u) {
        apply_action_write(&current_state, action_index, params.gridSize);
    }

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

    for (var i2: u32 = 0; i2 < state_size; i2 = i2 + 1) {
      current_state[i2] = next_state[i2];
    }
  }

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
  let out_offset = batch_idx * state_size;
  for (var i3: u32 = 0; i3 < state_size; i3 = i3 + 1) {
    output[out_offset + i3] = current_state[i3];
  }
}
`;

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

function applyAction(state: Uint8Array, actionIndex: number, agentPos: {x: number, y: number}): Uint8Array {
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
        const yy = (agentPos.y + i + size) % size;
        const xx = (agentPos.x + j + size) % size;
        const idx = yy * size + xx;
        newState[idx] = patternBits[i * 2 + j];
      }
    }
  }
  return newState;
}

function evaluateSequenceToGrid(seq: Uint32Array | null): Uint8Array | null {
  if (!seq || !currentInitialState) return null;
  const size = Math.sqrt(currentInitialState.length);
  let state: Uint8Array = new Uint8Array(currentInitialState);
  let agent = { x: size >> 1, y: size >> 1 };
  for (let a = 0; a < seq.length; a++) {
    const actionIndex = seq[a];
    state = applyAction(state, actionIndex, agent);
    state = conwayStep(state);
  }
  return state;
}

function renderGrid(canvas: HTMLCanvasElement, grid: Uint8Array | null, cellColor: string = '#2196F3', showAgent: boolean = false, agentX: number = 0, agentY: number = 0) {
  const size = Math.sqrt(grid ? grid.length : GRID_SIZE * GRID_SIZE);
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const cell = canvas.width / size;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = '#ddd';
  ctx.fillStyle = cellColor;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      if (grid && grid[y * size + x] === 1) {
        ctx.fillRect(Math.floor(x * cell), Math.floor(y * cell), Math.ceil(cell), Math.ceil(cell));
      }
      ctx.strokeRect(Math.floor(x * cell), Math.floor(y * cell), Math.ceil(cell), Math.ceil(cell));
    }
  }

  if (showAgent) {
    ctx.strokeStyle = '#FF00FF';
    ctx.lineWidth = 3;
    const px = agentX * cell;
    const py = agentY * cell;
    ctx.strokeRect(px, py, 2 * cell, 2 * cell);
    ctx.lineWidth = 1;
  }
}

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
    manualResetBtn: document.getElementById('manualResetBtn') as HTMLButtonElement,
    actionUsageCanvas: document.getElementById('actionUsageCanvas') as HTMLCanvasElement,
    fitnessDistCanvas: document.getElementById('fitnessDistCanvas') as HTMLCanvasElement,
  };
}

function initEvolveMode(ui: UIElements): void {
  if (sharedTargetPattern) {
    renderGrid(ui.targetVizCanvas, sharedTargetPattern, '#F44336');
  }
}

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

function calculateGridStats(grid: Uint8Array): { density: number } {
  const aliveCells = grid.reduce((a, b) => a + b, 0);
  const density = aliveCells / grid.length;
  return { density };
}

function updateManualDisplay() {
  const ui = getUIElements();
  renderGrid(ui.manualDemoCanvas, manualDemoState, '#2196F3', true, manualAgentX, manualAgentY);
  renderGrid(ui.manualTargetCanvas, manualTargetState, '#F44336', false);

  const stats = calculateGridStats(manualDemoState);
  ui.manualStats.innerHTML = `
    Step: <strong>${manualStep}/${MANUAL_MAX_STEPS}</strong><br>
    Agent Position: <strong>(${manualAgentX}, ${manualAgentY})</strong><br>
    Density: <strong>${stats.density.toFixed(2)}</strong>
  `;
}

function initManualMode(ui: UIElements) {
  ui.manualDemoCanvas.width = 240;
  ui.manualDemoCanvas.height = 240;
  ui.manualTargetCanvas.width = 240;
  ui.manualTargetCanvas.height = 240;

  const cell = ui.manualTargetCanvas.width / GRID_SIZE;
  
  const targetClickHandler = (e: MouseEvent) => {
    const rect = ui.manualTargetCanvas.getBoundingClientRect();
    const x = Math.floor((e.clientX - rect.left) / cell);
    const y = Math.floor((e.clientY - rect.top) / cell);
    if (x >= 0 && x < GRID_SIZE && y >= 0 && y < GRID_SIZE) {
      const idx = y * GRID_SIZE + x;
      manualTargetState[idx] = 1 - manualTargetState[idx];
      sharedTargetPattern = new Uint8Array(manualTargetState);
      updateManualDisplay();
    }
  };

  ui.manualTargetCanvas.removeEventListener('click', targetClickHandler);
  ui.manualTargetCanvas.addEventListener('click', targetClickHandler);

  ui.savePatternBtn.addEventListener('click', () => {
    manualTargetState = new Uint8Array(manualDemoState);
    sharedTargetPattern = new Uint8Array(manualTargetState);
    log(ui.log, 'Pattern saved from demo canvas to target (shared).');
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

  document.addEventListener('keydown', (e) => {
    if (currentMode !== 'manual') return;
    if (manualStep >= MANUAL_MAX_STEPS) return;

    const key = e.key.toLowerCase();

    if (key === 'arrowup' || key === 'arrowdown' || key === 'arrowleft' || key === 'arrowright') {
      e.preventDefault();
      if (key === 'arrowup') manualAgentY = (manualAgentY - 1 + GRID_SIZE) % GRID_SIZE;
      else if (key === 'arrowdown') manualAgentY = (manualAgentY + 1) % GRID_SIZE;
      else if (key === 'arrowleft') manualAgentX = (manualAgentX - 1 + GRID_SIZE) % GRID_SIZE;
      else if (key === 'arrowright') manualAgentX = (manualAgentX + 1 + GRID_SIZE) % GRID_SIZE;

      manualDemoState = conwayStep(manualDemoState);
      manualStep = Math.min(manualStep + 1, MANUAL_MAX_STEPS);
      updateManualDisplay();
    } else if (key === ' ') {
      e.preventDefault();
      manualDemoState = conwayStep(manualDemoState);
      manualStep = Math.min(manualStep + 1, MANUAL_MAX_STEPS);
      updateManualDisplay();
    } else if (/^[0-9a-f]$/.test(key)) {
      const patternNum = parseInt(key, 16);
      const bits = [(patternNum >> 3) & 1, (patternNum >> 2) & 1, (patternNum >> 1) & 1, patternNum & 1];
      for (let i = 0; i < 2; i++) {
        for (let j = 0; j < 2; j++) {
          const yy = (manualAgentY + i + GRID_SIZE) % GRID_SIZE;
          const xx = (manualAgentX + j + GRID_SIZE) % GRID_SIZE;
          const idx = yy * GRID_SIZE + xx;
          manualDemoState[idx] = bits[i * 2 + j];
        }
      }
      manualStep = Math.min(manualStep + 1, MANUAL_MAX_STEPS);
      updateManualDisplay();
    } else if (key === 's') {
      manualTargetState = new Uint8Array(manualDemoState);
      sharedTargetPattern = new Uint8Array(manualTargetState);
      log(ui.log, 'Pattern saved from demo canvas to target (shared).');
      updateManualDisplay();
    }
  });

  updateManualDisplay();
}

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

  const center = Math.floor(GRID_SIZE / 2);
  const initial = new Uint32Array(GRID_SIZE * GRID_SIZE);
  initial[center * GRID_SIZE + center] = 1;
  initial[center * GRID_SIZE + center + 1] = 1;
  initial[(center + 1) * GRID_SIZE + center] = 1;
  initial[(center + 1) * GRID_SIZE + center + 1] = 1;
  currentInitialState = new Uint8Array(initial.length);
  for (let i = 0; i < initial.length; i++) currentInitialState[i] = initial[i];

  const device = await requestDevice();
  const batchSize = Number(ui.population.value) || 50;
  const steps = Number(ui.steps.value) || 12;
  const generations = Number(ui.generations.value) || 100;
  const mutationRate = Number(ui.mut.value) || 0.1;
  const eliteFrac = Number(ui.elite.value) || 0.2;
  const vizFreq = Number(ui.vizFreq.value) || 5;

  await setupComputePipeline(batchSize, steps);

  const stateSize = GRID_SIZE * GRID_SIZE;
  const inputStatesSize = batchSize * stateSize;
  const inputSequenceSize = batchSize * steps;
  const inputTotalSize = (inputStatesSize + inputSequenceSize);
  const inputBufferSizeBytes = inputTotalSize * 4;

  const inputBuffer = device.createBuffer({
    size: inputBufferSizeBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  const outputBuffer = device.createBuffer({
    size: inputStatesSize * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const targetBuffer = device.createBuffer({
    size: stateSize * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    mappedAtCreation: false,
  });
  const paddedTarget = new Uint32Array(stateSize);
  for (let i = 0; i < stateSize; i++) paddedTarget[i] = (sharedTargetPattern ? sharedTargetPattern[i] : 0);
  device.queue.writeBuffer(targetBuffer, 0, paddedTarget.buffer, paddedTarget.byteOffset, paddedTarget.byteLength);

  const fitnessBuffer = device.createBuffer({
    size: batchSize * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const fitnessReadBuffer = device.createBuffer({
    size: batchSize * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const outputReadBuffer = device.createBuffer({
    size: inputStatesSize * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const paramsBufferSize = 4 * 4;
  const paramsBuffer = device.createBuffer({
    size: paramsBufferSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const computePipeline = pipeline!;
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

  function randomSequenceArray(pop: number, stepsCount: number): Uint32Array {
    const arr = new Uint32Array(pop * stepsCount);
    for (let i = 0; i < arr.length; i++) {
      arr[i] = Math.floor(Math.random() * ACTIONS.length);
    }
    return arr;
  }

  let populationSequences = randomSequenceArray(batchSize, steps);
  let inputStates = new Uint32Array(inputStatesSize);
  for (let b = 0; b < batchSize; b++) {
    const base = b * stateSize;
    for (let i = 0; i < stateSize; i++) {
      inputStates[base + i] = initial[i];
    }
  }

  const inputFullArray = new Uint32Array(inputTotalSize);
  inputFullArray.set(inputStates, 0);
  inputFullArray.set(populationSequences, inputStatesSize);

  device.queue.writeBuffer(inputBuffer, 0, inputFullArray.buffer, inputFullArray.byteOffset, inputFullArray.byteLength);

  const paramsArray = new Uint32Array([GRID_SIZE, batchSize, steps, 0]);
  device.queue.writeBuffer(paramsBuffer, 0, paramsArray.buffer, paramsArray.byteOffset, paramsArray.byteLength);

  ui.statsDiv.innerHTML = `Gen: 0<br>Best Fitness: N/A<br>Avg Fitness: N/A<br>Elite: ${eliteFrac}<br>Mut rate: ${mutationRate}`;

  isRunning = true;
  currentGeneration = 0;
  bestFitness = Infinity;
  bestSequence = null;
  avgFitnessHistory = [];
  maxFitnessHistory = [];

  const seqSet = new Set<string>();

  for (let gen = 0; gen < generations && isRunning; gen++) {
    currentGeneration = gen;

    paramsArray[3] = 0;
    device.queue.writeBuffer(paramsBuffer, 0, paramsArray.buffer, paramsArray.byteOffset, paramsArray.byteLength);

    inputFullArray.set(inputStates, 0);
    inputFullArray.set(populationSequences, inputStatesSize);
    device.queue.writeBuffer(inputBuffer, 0, inputFullArray.buffer, inputFullArray.byteOffset, inputFullArray.byteLength);

    const commandEncoder = device.createCommandEncoder();
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(computePipeline);
    pass.setBindGroup(0, bindGroup!);
    const workgroupCount = Math.ceil(batchSize / 64);
    pass.dispatchWorkgroups(workgroupCount);
    pass.end();
    commandEncoder.copyBufferToBuffer(fitnessBuffer, 0, fitnessReadBuffer, 0, batchSize * 4);
    commandEncoder.copyBufferToBuffer(outputBuffer, 0, outputReadBuffer, 0, inputStatesSize * 4);
    device.queue.submit([commandEncoder.finish()]);

    await fitnessReadBuffer.mapAsync(GPUMapMode.READ);
    const fitnessArrayBuffer = fitnessReadBuffer.getMappedRange();
    const fitnessArray = new Uint32Array(fitnessArrayBuffer.slice(0));
    fitnessReadBuffer.unmap();

    await outputReadBuffer.mapAsync(GPUMapMode.READ);
    const outputArrayBuffer = outputReadBuffer.getMappedRange();
    const outputArray = new Uint32Array(outputArrayBuffer.slice(0));
    outputReadBuffer.unmap();

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

    if (minFitness < bestFitness) {
      bestFitness = minFitness;
      const seq = new Uint32Array(steps);
      for (let s = 0; s < steps; s++) {
        seq[s] = populationSequences[minIdx * steps + s];
      }
      bestSequence = seq;
      log(ui.log, `Gen ${gen}: New best fitness ${bestFitness} (idx ${minIdx})`);
      const bestGrid = evaluateSequenceToGrid(bestSequence);
      if (bestGrid && ui.evolveBestCanvas) {
        renderGrid(ui.evolveBestCanvas, bestGrid, '#4CAF50', false);
      }
    }

    seqSet.clear();
    const sampleCount = Math.min(20, batchSize);
    for (let s = 0; s < sampleCount; s++) {
      const idx = Math.floor(Math.random() * batchSize);
      const seqStr = Array.from(populationSequences.slice(idx * steps, (idx + 1) * steps)).join(',');
      seqSet.add(seqStr);
    }
    const diversity = seqSet.size / sampleCount;

    avgFitnessHistory.push(avgFitness);
    maxFitnessHistory.push(minFitness);

    ui.statsDiv.innerHTML = `
      Gen: <strong>${gen}</strong><br>
      Best Fitness: <strong>${bestFitness === Infinity ? 'N/A' : bestFitness}</strong><br>
      Avg Fitness: <strong>${avgFitness.toFixed(2)}</strong><br>
      Population: <strong>${batchSize}</strong><br>
      Steps: <strong>${steps}</strong><br>
      Diversity (sample): <strong>${diversity.toFixed(2)}</strong><br>
      Mutation rate: <strong>${mutationRate}</strong>
    `;

    if (ui.targetVizCanvas) renderGrid(ui.targetVizCanvas, sharedTargetPattern ? new Uint8Array(sharedTargetPattern) : new Uint8Array(stateSize), '#F44336');

    if (ui.evolveCurrentCanvas) {
      const sampleOut = new Uint8Array(stateSize);
      for (let i = 0; i < stateSize; i++) sampleOut[i] = outputArray[i];
      renderGrid(ui.evolveCurrentCanvas, sampleOut, '#2196F3', false);
    }

    if (gen % vizFreq === 0 || gen === generations - 1) {
      renderFitnessChart(ui.fitnessDistCanvas, avgFitnessHistory, maxFitnessHistory);
      renderActionUsage(ui.actionUsageCanvas, populationSequences, bestSequence, steps);
    }

    if (bestFitness === 0) {
      log(ui.log, `Perfect match found in generation ${gen}.`);
      break;
    }

    const indices = new Array(batchSize).fill(0).map((_, i) => i);
    indices.sort((a, b) => fitnessArray[a] - fitnessArray[b]);

    const eliteCount = Math.max(1, Math.floor(batchSize * eliteFrac));
    const newPopulation = new Uint32Array(batchSize * steps);
    for (let e = 0; e < eliteCount; e++) {
      const srcIdx = indices[e];
      newPopulation.set(populationSequences.slice(srcIdx * steps, srcIdx * steps + steps), e * steps);
    }
    for (let i = eliteCount; i < batchSize; i++) {
      const a = indices[Math.floor(Math.random() * Math.max(1, batchSize * 0.5))];
      const b = indices[Math.floor(Math.random() * Math.max(1, batchSize * 0.5))];
      const parentA = populationSequences.slice(a * steps, a * steps + steps);
      const parentB = populationSequences.slice(b * steps, b * steps + steps);
      const child = new Uint32Array(steps);
      const crossPoint = Math.floor(Math.random() * steps);
      for (let s = 0; s < steps; s++) {
        child[s] = (s < crossPoint) ? parentA[s] : parentB[s];
        if (Math.random() < mutationRate) {
          child[s] = Math.floor(Math.random() * ACTIONS.length);
        }
      }
      newPopulation.set(child, i * steps);
    }

    populationSequences = newPopulation;

    if (gen % vizFreq === 0) {
      await new Promise(r => setTimeout(r, 1));
    }
  }

  if (bestSequence) {
    const bestGrid = evaluateSequenceToGrid(bestSequence);
    if (bestGrid && ui.evolveBestCanvas) renderGrid(ui.evolveBestCanvas, bestGrid, '#4CAF50', false);
  }

  isRunning = false;
  ui.startBtn.disabled = false;
  ui.stopBtn.disabled = true;

  log(ui.log, 'Evolution complete. Best sequence available for Demo mode.');
}

function renderFitnessChart(canvas: HTMLCanvasElement, avgHistory: number[], maxHistory: number[]) {
  const ctx = canvas.getContext('2d');
  if (!ctx || avgHistory.length === 0) return;

  const width = canvas.width;
  const height = canvas.height;
  const padding = 40;

  ctx.clearRect(0, 0, width, height);

  const maxGen = avgHistory.length;
  const maxFit = Math.max(...avgHistory, ...maxHistory);
  const minFit = Math.min(...avgHistory, ...maxHistory);
  const fitRange = maxFit - minFit || 1;

  ctx.strokeStyle = '#ddd';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const y = padding + (height - 2 * padding) * i / 5;
    ctx.beginPath();
    ctx.moveTo(padding, y);
    ctx.lineTo(width - padding, y);
    ctx.stroke();
  }

  ctx.strokeStyle = '#4CAF50';
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < maxHistory.length; i++) {
    const x = padding + (width - 2 * padding) * i / maxGen;
    const y = height - padding - (height - 2 * padding) * (maxHistory[i] - minFit) / fitRange;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  ctx.strokeStyle = '#2196F3';
  ctx.beginPath();
  for (let i = 0; i < avgHistory.length; i++) {
    const x = padding + (width - 2 * padding) * i / maxGen;
    const y = height - padding - (height - 2 * padding) * (avgHistory[i] - minFit) / fitRange;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  ctx.fillStyle = '#333';
  ctx.font = '12px Arial';
  ctx.textAlign = 'center';
  ctx.fillText('Generation', width / 2, height - 5);
  ctx.save();
  ctx.translate(10, height / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('Fitness', 0, 0);
  ctx.restore();
}

function renderActionUsage(canvas: HTMLCanvasElement, population: Uint32Array, bestSeq: Uint32Array | null, steps: number) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const width = canvas.width;
  const height = canvas.height;
  const numActions = ACTIONS.length;

  ctx.clearRect(0, 0, width, height);

  const actionCounts = new Array(numActions).fill(0);
  for (let i = 0; i < population.length; i++) {
    actionCounts[population[i]]++;
  }

  const maxCount = Math.max(...actionCounts) || 1;
  const barWidth = width / numActions;

  ctx.fillStyle = '#2196F3';
  for (let i = 0; i < numActions; i++) {
    const barHeight = (actionCounts[i] / maxCount) * (height - 20);
    ctx.fillRect(i * barWidth, height - barHeight, barWidth - 2, barHeight);
  }

  if (bestSeq) {
    ctx.fillStyle = 'rgba(255, 152, 0, 0.7)';
    const bestCounts = new Array(numActions).fill(0);
    for (let i = 0; i < bestSeq.length; i++) {
      bestCounts[bestSeq[i]]++;
    }
    for (let i = 0; i < numActions; i++) {
      const barHeight = (bestCounts[i] / steps) * (height - 20);
      ctx.fillRect(i * barWidth, height - barHeight, barWidth - 2, barHeight);
    }
  }
}

function updateDemoDisplay(ui: UIElements) {
  renderGrid(ui.demoTargetCanvas, sharedTargetPattern ? new Uint8Array(sharedTargetPattern) : new Uint8Array(GRID_SIZE * GRID_SIZE), '#F44336');

  if (demoPlaybackState && bestSequence) {
    renderGrid(ui.demoBestCanvas, demoPlaybackState, '#2196F3', true, demoAgentX, demoAgentY);
    ui.demoStep.textContent = demoPlaybackStep.toString();

    const seqCanvas = ui.demoActionsCanvas;
    const ctx = seqCanvas.getContext('2d');
    if (ctx) {
      const steps = bestSequence.length;
      const itemW = seqCanvas.width / steps;

      ctx.clearRect(0, 0, seqCanvas.width, seqCanvas.height);
      
      const actionLabels = ['↑', '↓', '←', '→', '∅'];
      for (let i = 0; i < 16; i++) {
        actionLabels.push(i.toString(16).toUpperCase());
      }

      for (let i = 0; i < steps; i++) {
        const a = bestSequence[i];
        if (i < demoPlaybackStep) ctx.fillStyle = '#ddd';
        else if (i === demoPlaybackStep) ctx.fillStyle = 'red';
        else ctx.fillStyle = '#f5f5f5';
        ctx.fillRect(i * itemW, 0, Math.ceil(itemW), seqCanvas.height);
        ctx.strokeStyle = '#999';
        ctx.strokeRect(i * itemW, 0, Math.ceil(itemW), seqCanvas.height);
        
        ctx.fillStyle = '#000';
        ctx.font = '10px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(actionLabels[a] || '?', i * itemW + itemW / 2, seqCanvas.height / 2);
      }
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

  log(ui.log, 'Ready. Draw or import a target pattern by toggling cells, or switch to Manual mode to craft one.');
}

initApp();