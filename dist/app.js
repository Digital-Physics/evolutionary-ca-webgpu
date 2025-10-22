"use strict";
// src/app.ts
// WebGPU CA compute + GPU fitness + genetic algorithm
// Modes: manual, evolve, demo
// --- CONSTANTS AND CONFIGURATION ---
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
// --- GLOBAL STATE ---
let currentMode = 'evolve';
// Manual Mode State
let manualDemoState = new Uint8Array(GRID_SIZE * GRID_SIZE);
let manualTargetState = new Uint8Array(GRID_SIZE * GRID_SIZE);
let manualStep = 0;
let manualAgentX = 5; // Start at (5,5) - upper left of center 2x2
let manualAgentY = 5;
// Shared state between modes
let sharedTargetPattern = null;
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
// --- COMPUTE SHADER (WGSL) ---
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

fn apply_action_move(state: ptr<function, array<u32, 144>>, action_index: u32) {
  var ax = (*state)[140u];
  var ay = (*state)[141u];
  let grid_size = params.gridSize;

  if (action_index == 0u) { ay = (ay + grid_size - 1u) % grid_size; }
  else if (action_index == 1u) { ay = (ay + 1u) % grid_size; }
  else if (action_index == 2u) { ax = (ax + grid_size - 1u) % grid_size; }
  else if (action_index == 3u) { ax = (ax + 1u) % grid_size; }
  
  (*state)[140u] = ax;
  (*state)[141u] = ay;
}

fn apply_action_write(state: ptr<function, array<u32, 144>>, action_index: u32) {
  let pat = action_index - 5u;
  let ax = (*state)[140u];
  let ay = (*state)[141u];
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

  var current_state: array<u32, 144>;
  for (var i: u32 = 0; i < grid_cell_count; i = i + 1) {
    current_state[i] = inputStates[state_offset + i];
  }
  current_state[140u] = 5u;
  current_state[141u] = 5u;

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

  fitness[idx] = (matches * 100u) / grid_cell_count;
  
  for (var i3: u32 = 0; i3 < grid_cell_count; i3 = i3 + 1) {
    outputStates[state_offset + i3] = current_state[i3];
  }
}
`;
// --- CPU-SIDE SIMULATION (for Manual/Demo modes) ---
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
// --- RENDERING ---
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
        clearPatternBtn: document.getElementById('clearPatternBtn'),
        savePatternBtn: document.getElementById('savePatternBtn'),
        demoTargetCanvas: document.getElementById('demoTargetCanvas'),
        demoBestCanvas: document.getElementById('demoBestCanvas'),
        demoActionsCanvas: document.getElementById('demoActionsCanvas'),
        demoStep: document.getElementById('demoStep'),
        demoMaxSteps: document.getElementById('demoMaxSteps'),
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
    };
}
// --- MODE HANDLING ---
function handleModeChange() {
    document.querySelectorAll('.mode-content').forEach(el => el.style.display = 'none');
    const selectedModeEl = document.getElementById(`${currentMode}-mode`);
    if (selectedModeEl) {
        selectedModeEl.style.display = 'block';
    }
    const ui = getUIElements();
    if (currentMode === 'manual') {
        renderGrid(ui.manualTargetCanvas, sharedTargetPattern, '#c53030');
        updateManualDisplay();
    }
    else if (currentMode === 'evolve') {
        renderGrid(ui.targetVizCanvas, sharedTargetPattern, '#c53030');
    }
    else if (currentMode === 'demo') {
        initializeDemoMode();
    }
}
// --- MANUAL MODE ---
function updateManualDisplay() {
    const ui = getUIElements();
    const maxSteps = Number(ui.steps.value);
    renderGrid(ui.manualDemoCanvas, manualDemoState, '#4299e1', true, manualAgentX, manualAgentY);
    renderGrid(ui.manualTargetCanvas, sharedTargetPattern, '#c53030');
    ui.manualStats.innerHTML = `
    Step: <strong>${manualStep}/${maxSteps}</strong><br>
    Agent Position: <strong>(${manualAgentX}, ${manualAgentY})</strong>
  `;
}
function initManualMode(ui) {
    const updateMaxSteps = () => {
        ui.manualMaxStepsLabel.textContent = ui.steps.value;
    };
    updateMaxSteps();
    ui.steps.addEventListener('change', updateMaxSteps);
    const targetClickHandler = (e) => {
        const canvas = e.currentTarget;
        const rect = canvas.getBoundingClientRect();
        const cell = canvas.width / GRID_SIZE;
        const x = Math.floor((e.clientX - rect.left) / cell);
        const y = Math.floor((e.clientY - rect.top) / cell);
        if (x >= 0 && x < GRID_SIZE && y >= 0 && y < GRID_SIZE) {
            const idx = y * GRID_SIZE + x;
            if (!sharedTargetPattern)
                sharedTargetPattern = new Uint8Array(GRID_SIZE * GRID_SIZE);
            sharedTargetPattern[idx] = 1 - sharedTargetPattern[idx];
            updateManualDisplay();
        }
    };
    ui.manualTargetCanvas.addEventListener('click', targetClickHandler);
    ui.savePatternBtn.addEventListener('click', () => {
        sharedTargetPattern = new Uint8Array(manualDemoState);
        log(ui.log, 'Saved manual canvas to shared target pattern.');
        updateManualDisplay();
    });
    ui.clearPatternBtn.addEventListener('click', () => {
        sharedTargetPattern = new Uint8Array(GRID_SIZE * GRID_SIZE);
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
    updateManualDisplay();
}
function handleManualKeyDown(e) {
    if (currentMode !== 'manual')
        return;
    const ui = getUIElements();
    const maxSteps = Number(ui.steps.value);
    if (manualStep >= maxSteps)
        return;
    const key = e.key.toLowerCase();
    let actionTaken = false;
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
        const agentPos = { x: manualAgentX, y: manualAgentY };
        manualDemoState = new Uint8Array(conwayStep(manualDemoState));
        manualDemoState = new Uint8Array(applyAction(manualDemoState, actionIndex, agentPos));
        manualAgentX = agentPos.x;
        manualAgentY = agentPos.y;
        manualStep++;
        actionTaken = true;
    }
    if (actionTaken) {
        updateManualDisplay();
    }
}
// --- LOGGING ---
function log(el, ...args) {
    if (!el)
        return;
    el.textContent += args.join(' ') + '\n';
    el.scrollTop = el.scrollHeight;
}
// --- WEBGPU SETUP ---
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
    const labels = ['↑', '↓', '←', '→', '∅', ...Array.from({ length: 16 }, (_, i) => (15 - i).toString(16).toUpperCase())];
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
    html += '<strong>TOP 5 SEQUENCES</strong><br>';
    html += '═══════════════════════<br>';
    if (top5Sequences.length === 0) {
        html += 'No sequences yet<br>';
    }
    else {
        top5Sequences.forEach((entry, idx) => {
            const seqStr = formatSequence(entry.sequence, 12);
            html += `#${idx + 1}: ${entry.fitness.toFixed(1)}% (Gen ${entry.generation})<br>`;
            html += `&nbsp;&nbsp;&nbsp;${seqStr}<br>`;
        });
    }
    html += '</div>';
    ui.top5List.innerHTML = html;
}
// --- EVOLUTION MODE ---
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
    const batchSize = Number(ui.population.value);
    const steps = Number(ui.steps.value);
    const generations = Number(ui.generations.value);
    const mutationRate = Number(ui.mut.value);
    const eliteFrac = Number(ui.elite.value);
    const vizFreq = Number(ui.vizFreq.value);
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
    const initialStates = new Uint32Array(batchSize * stateSize);
    device.queue.writeBuffer(inputStatesBuffer, 0, initialStates);
    const targetPattern32 = new Uint32Array(sharedTargetPattern);
    device.queue.writeBuffer(targetBuffer, 0, targetPattern32);
    const paramsArray = new Uint32Array([GRID_SIZE, batchSize, steps]);
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
    let populationSequences = new Uint32Array(batchSize * steps).map(() => Math.floor(Math.random() * ACTIONS.length));
    bestFitness = 0;
    bestSequence = null;
    avgFitnessHistory = [];
    allTimeBestFitnessHistory = [];
    diversityHistory = [];
    top5Sequences = [];
    currentGeneration = 0;
    for (let gen = 0; gen < generations && isRunning; gen++) {
        currentGeneration = gen;
        device.queue.writeBuffer(inputSequencesBuffer, 0, populationSequences);
        const commandEncoder = device.createCommandEncoder();
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(pipeline);
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
        // Update top 5 sequences
        const sortedIndices = Array.from({ length: batchSize }, (_, i) => i).sort((a, b) => fitnessArray[b] - fitnessArray[a]);
        for (let i = 0; i < Math.min(5, batchSize); i++) {
            const idx = sortedIndices[i];
            const seq = populationSequences.slice(idx * steps, (idx + 1) * steps);
            const fit = fitnessArray[idx];
            const seqKey = Array.from(seq).join(',');
            const exists = top5Sequences.find(e => Array.from(e.sequence).join(',') === seqKey);
            if (!exists && fit > 0) {
                top5Sequences.push({
                    sequence: new Uint32Array(seq),
                    fitness: fit,
                    generation: gen
                });
            }
        }
        top5Sequences.sort((a, b) => b.fitness - a.fitness);
        top5Sequences = top5Sequences.slice(0, 5);
        const seqSet = new Set();
        for (let i = 0; i < batchSize; i++) {
            seqSet.add(populationSequences.slice(i * steps, (i + 1) * steps).join(','));
        }
        diversityHistory.push(seqSet.size / batchSize);
        avgFitnessHistory.push(avgFitness);
        allTimeBestFitnessHistory.push(bestFitness);
        if (gen % vizFreq === 0 || gen === generations - 1 || bestFitness === 100) {
            const bestSeqStr = bestSequence ? formatSequence(bestSequence, 20) : 'N/A';
            ui.statsDiv.innerHTML = `
        Gen: <strong>${gen} / ${generations}</strong><br>
        Best Fitness: <strong>${bestFitness}%</strong><br>
        Avg Fitness: <strong>${avgFitness.toFixed(1)}%</strong><br>
        <hr style="margin: 4px 0; border-top: 1px solid #e2e8f0;">
        Best Sequence: <small>${bestSeqStr}</small>
      `;
            renderGrid(ui.evolveBestCanvas, evaluateSequenceToGrid(bestSequence), '#38a169');
            renderGrid(ui.targetVizCanvas, sharedTargetPattern, '#c53030');
            // Render random sample
            const randomIdx = Math.floor(Math.random() * batchSize);
            const randomSeq = populationSequences.slice(randomIdx * steps, (randomIdx + 1) * steps);
            renderGrid(ui.evolveCurrentCanvas, evaluateSequenceToGrid(randomSeq), '#4299e1');
            renderFitnessChart(ui.fitnessDistCanvas, avgFitnessHistory, allTimeBestFitnessHistory);
            renderDiversityChart(ui.diversityCanvas, diversityHistory);
            updateTop5Display(ui);
            await new Promise(r => setTimeout(r, 1));
        }
        if (bestFitness === 100) {
            log(ui.log, `Perfect match found in generation ${gen}!`);
            break;
        }
        const sortedIndices2 = Array.from({ length: batchSize }, (_, i) => i).sort((a, b) => fitnessArray[b] - fitnessArray[a]);
        const newPopulation = new Uint32Array(batchSize * steps);
        const eliteCount = Math.floor(batchSize * eliteFrac);
        for (let i = 0; i < eliteCount; i++) {
            const bestIdx = sortedIndices2[i];
            newPopulation.set(populationSequences.slice(bestIdx * steps, (bestIdx + 1) * steps), i * steps);
        }
        for (let i = eliteCount; i < batchSize; i++) {
            const parentA_idx = sortedIndices2[Math.floor(Math.random() * batchSize * 0.5)];
            const parentB_idx = sortedIndices2[Math.floor(Math.random() * batchSize * 0.5)];
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
function renderChart(canvas, datasets, yLabel, xLabel, yMax) {
    const ctx = canvas.getContext('2d');
    if (!ctx || datasets.length === 0 || datasets[0].data.length === 0)
        return;
    const { width, height } = canvas;
    const p = { t: 30, r: 20, b: 40, l: 50 };
    ctx.clearRect(0, 0, width, height);
    ctx.font = '12px sans-serif';
    const xRange = width - p.l - p.r;
    const yRange = height - p.t - p.b;
    const numPoints = datasets[0].data.length;
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
    datasets.forEach(({ data, color }) => {
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let i = 0; i < numPoints; i++) {
            const x = p.l + (xRange * i) / Math.max(1, numPoints - 1);
            const y = p.t + yRange * (1 - data[i] / yMax);
            if (i === 0)
                ctx.moveTo(x, y);
            else
                ctx.lineTo(x, y);
        }
        ctx.stroke();
    });
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
    ], 'Fitness %', 'Generation', 100);
}
function renderDiversityChart(canvas, diversity) {
    renderChart(canvas, [
        { data: diversity, color: '#9f7aea', label: 'Diversity' }
    ], 'Diversity Ratio', 'Generation', 1);
}
// --- DEMO MODE ---
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
function updateDemoDisplay(ui) {
    renderGrid(ui.demoTargetCanvas, sharedTargetPattern, '#c53030');
    if (demoPlaybackState && bestSequence) {
        renderGrid(ui.demoBestCanvas, demoPlaybackState, '#4299e1', true, demoAgentX, demoAgentY);
        ui.demoStep.textContent = demoPlaybackStep.toString();
        const container = ui.demoActionsCanvas;
        container.innerHTML = '';
        container.style.display = 'flex';
        container.style.flexWrap = 'wrap';
        container.style.gap = '4px';
        container.style.padding = '8px';
        container.style.backgroundColor = '#f7fafc';
        container.style.border = '1px solid #e2e8f0';
        container.style.borderRadius = '0.375rem';
        const actionLabels = ['↑', '↓', '←', '→', 'Space', ...Array.from({ length: 16 }, (_, i) => (15 - i).toString(16).toUpperCase())];
        for (let i = 0; i < bestSequence.length; i++) {
            const wrapper = document.createElement('div');
            wrapper.style.display = 'flex';
            wrapper.style.flexDirection = 'column';
            wrapper.style.alignItems = 'center';
            wrapper.style.gap = '2px';
            if (i === demoPlaybackStep) {
                wrapper.style.backgroundColor = '#f6ad55';
                wrapper.style.padding = '2px';
                wrapper.style.borderRadius = '4px';
            }
            const canvas = createActionImage(bestSequence[i], 40);
            wrapper.appendChild(canvas);
            const label = document.createElement('div');
            label.textContent = actionLabels[bestSequence[i]];
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
        demoPlaybackState = new Uint8Array(GRID_SIZE * GRID_SIZE);
    }
    else {
        log(ui.log, `Loaded sequence with ${bestSequence.length} steps for demo.`);
        ui.demoMaxSteps.textContent = bestSequence.length.toString();
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
// --- APP INITIALIZATION ---
function initApp() {
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
        const cell = canvas.width / GRID_SIZE;
        const x = Math.floor((e.clientX - rect.left) / cell);
        const y = Math.floor((e.clientY - rect.top) / cell);
        if (x >= 0 && x < GRID_SIZE && y >= 0 && y < GRID_SIZE) {
            const idx = y * GRID_SIZE + x;
            sharedTargetPattern[idx] = 1 - sharedTargetPattern[idx];
            renderGrid(canvas, sharedTargetPattern, '#c53030');
        }
    };
    ui.targetVizCanvas.addEventListener('click', targetClickHandler);
    ui.demoTargetCanvas.addEventListener('click', targetClickHandler);
    initManualMode(ui);
    handleModeChange();
    log(ui.log, 'App ready. Draw a target pattern, then start evolution.');
}
window.addEventListener('DOMContentLoaded', initApp);
