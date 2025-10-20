// src/app.ts
// WebGPU CA compute + GPU fitness + genetic algorithm
// Modes: manual, evolve, demo
//
// Updated: fixed demo playback, manual reset, stats, seq strip, and TypeScript typing fixes.

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
};

const GRID_SIZE = 12;
const MAX_GENERATIONS_DEFAULT = 200;

function log(el: HTMLElement, ...args: any[]) {
  el.textContent += args.join(' ') + '\n';
  el.scrollTop = el.scrollHeight;
}

async function requestDevice(): Promise<GPUDevice> {
  if (!navigator.gpu) throw new Error('WebGPU not supported');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No GPU adapter found');
  return await adapter.requestDevice();
}

// ---------- WGSL compute shader ----------
const COMPUTE_SHADER = `
struct Params {
  gridSize: u32,
  batchSize: u32,
  steps: u32,
  currentStep: u32,
  isLastStep: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> srcGrid: array<u32>;
@group(0) @binding(2) var<storage, read_write> dstGrid: array<u32>;
@group(0) @binding(3) var<storage, read> actions: array<u32>;
@group(0) @binding(4) var<storage, read> targetGrid: array<u32>;
@group(0) @binding(5) var<storage, read_write> fitnessCounts: array<atomic<u32>>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let g = gid.z;
  let size = params.gridSize;

  if (x >= size || y >= size || g >= params.batchSize) { return; }

  let gridIndex = g * size * size + y * size + x;
  
  // Count neighbors
  var count: u32 = 0u;
  for (var dy: i32 = -1; dy <= 1; dy++) {
    for (var dx: i32 = -1; dx <= 1; dx++) {
      if (dx == 0 && dy == 0) { continue; }
      let nx = (i32(x) + dx + i32(size)) % i32(size);
      let ny = (i32(y) + dy + i32(size)) % i32(size);
      let nidx = g * size * size + u32(ny) * size + u32(nx);
      count += srcGrid[nidx];
    }
  }

  let alive = srcGrid[gridIndex];
  var next: u32 = 0u;

  // Conway's Game of Life
  if (alive == 1u && (count == 2u || count == 3u)) {
    next = 1u;
  } else if (alive == 0u && count == 3u) {
    next = 1u;
  }

  // Apply action
  let actionIdx = g * params.steps + params.currentStep;
  let action = actions[actionIdx];

  if (action > 0u) {
    let atype = (action >> 24u) & 0xFFu;
    let pat = (action >> 16u) & 0xFFu;
    let pos = action & 0xFFFFu;

    if (atype == 2u) {
      let wy = pos / size;
      let wx = pos % size;
      
      if ((y >= wy && y < wy + 2u) && (x >= wx && x < wx + 2u)) {
        let ly = y - wy;
        let lx = x - wx;
        let bit = (pat >> (ly * 2u + lx)) & 1u;
        next = bit;
      }
    }
  }

  dstGrid[gridIndex] = next;

  if (params.isLastStep == 1u) {
    let tval = targetGrid[y * size + x];
    if (next == tval) {
      atomicAdd(&fitnessCounts[g], 1u);
    }
  }
}
`;

// ---------- GPUEvolver (same as before) ----------
class GPUEvolver {
  device: GPUDevice;
  pipeline: GPUComputePipeline;
  gridSize: number;
  batchSize: number;
  steps: number = 0;
  targetBuffer: GPUBuffer;
  gridBufferA: GPUBuffer;
  gridBufferB: GPUBuffer;
  actionsBuffer: GPUBuffer | null = null;
  fitnessBuffer: GPUBuffer;
  fitnessReadBuffer: GPUBuffer;
  paramsBuffer: GPUBuffer;
  zeroedGrid: Uint32Array;
  lastGridBuffer: GPUBuffer | null = null;

  constructor(device: GPUDevice, gridSize: number, batchSize: number) {
    this.device = device;
    this.gridSize = gridSize;
    this.batchSize = batchSize;

    const module = device.createShaderModule({ code: COMPUTE_SHADER });
    this.pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: 'main' }
    });

    const totalCells = batchSize * gridSize * gridSize;
    this.zeroedGrid = new Uint32Array(totalCells);

    const bufSize = totalCells * 4;
    this.gridBufferA = device.createBuffer({
      size: bufSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    });
    this.gridBufferB = device.createBuffer({
      size: bufSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    });
    device.queue.writeBuffer(this.gridBufferA, 0, this.zeroedGrid.buffer, this.zeroedGrid.byteOffset, this.zeroedGrid.byteLength);
    device.queue.writeBuffer(this.gridBufferB, 0, this.zeroedGrid.buffer, this.zeroedGrid.byteOffset, this.zeroedGrid.byteLength);

    this.targetBuffer = device.createBuffer({
      size: gridSize * gridSize * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    const emptyTarget = new Uint32Array(gridSize * gridSize);
    device.queue.writeBuffer(this.targetBuffer, 0, emptyTarget.buffer, emptyTarget.byteOffset, emptyTarget.byteLength);

    this.fitnessBuffer = device.createBuffer({
      size: batchSize * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });
    this.fitnessReadBuffer = device.createBuffer({
      size: batchSize * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    this.paramsBuffer = device.createBuffer({
      size: 20,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
  }

  setTarget(target: Uint32Array): void {
    this.device.queue.writeBuffer(this.targetBuffer, 0, target.buffer, target.byteOffset, target.byteLength);
  }

  async evaluate(actions: Uint32Array, steps: number): Promise<Float32Array> {
    if (actions.length !== this.batchSize * steps) {
      throw new Error(`Actions length mismatch: got ${actions.length}, expected ${this.batchSize * steps}`);
    }

    this.steps = steps;

    this.device.queue.writeBuffer(this.gridBufferA, 0, this.zeroedGrid.buffer, this.zeroedGrid.byteOffset, this.zeroedGrid.byteLength);
    this.device.queue.writeBuffer(this.gridBufferB, 0, this.zeroedGrid.buffer, this.zeroedGrid.byteOffset, this.zeroedGrid.byteLength);

    if (!this.actionsBuffer || this.actionsBuffer.size < actions.byteLength) {
      this.actionsBuffer = this.device.createBuffer({
        size: actions.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
      });
    }
    this.device.queue.writeBuffer(this.actionsBuffer, 0, actions.buffer, actions.byteOffset, actions.byteLength);

    const zeroFits = new Uint32Array(this.batchSize);
    this.device.queue.writeBuffer(this.fitnessBuffer, 0, zeroFits.buffer, zeroFits.byteOffset, zeroFits.byteLength);

    const wg = 8;
    const dispatchX = Math.ceil(this.gridSize / wg);
    const dispatchY = Math.ceil(this.gridSize / wg);
    const dispatchZ = this.batchSize;

    let srcBuffer = this.gridBufferA;
    let dstBuffer = this.gridBufferB;

    for (let step = 0; step < steps; step++) {
      const isLast = step === steps - 1 ? 1 : 0;
      const params = new Uint32Array([this.gridSize, this.batchSize, steps, step, isLast]);
      this.device.queue.writeBuffer(this.paramsBuffer, 0, params.buffer, params.byteOffset, params.byteLength);

      const bindGroup = this.device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.paramsBuffer } },
          { binding: 1, resource: { buffer: srcBuffer } },
          { binding: 2, resource: { buffer: dstBuffer } },
          { binding: 3, resource: { buffer: this.actionsBuffer! } },
          { binding: 4, resource: { buffer: this.targetBuffer } },
          { binding: 5, resource: { buffer: this.fitnessBuffer } }
        ]
      });

      const enc = this.device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
      pass.end();
      this.device.queue.submit([enc.finish()]);

      [srcBuffer, dstBuffer] = [dstBuffer, srcBuffer];
    }

    this.lastGridBuffer = srcBuffer;

    await this.device.queue.onSubmittedWorkDone();

    const encRead = this.device.createCommandEncoder();
    encRead.copyBufferToBuffer(this.fitnessBuffer, 0, this.fitnessReadBuffer, 0, this.batchSize * 4);
    this.device.queue.submit([encRead.finish()]);

    await this.fitnessReadBuffer.mapAsync(GPUMapMode.READ);
    const mapped = this.fitnessReadBuffer.getMappedRange();
    const fitU32 = new Uint32Array(mapped).slice();
    this.fitnessReadBuffer.unmap();

    const cellCount = this.gridSize * this.gridSize;
    const fitness = new Float32Array(this.batchSize);
    for (let i = 0; i < this.batchSize; i++) {
      fitness[i] = fitU32[i] / cellCount;
    }

    return fitness;
  }

  async getGrid(index: number): Promise<Uint8Array> {
    if (!this.lastGridBuffer) throw new Error('No evaluation yet');
    if (index < 0 || index >= this.batchSize) throw new Error('Index out of bounds');

    const cellsPerGrid = this.gridSize * this.gridSize;
    const bytesPerGrid = cellsPerGrid * 4;
    const offset = index * bytesPerGrid;

    const readBuf = this.device.createBuffer({
      size: bytesPerGrid,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.lastGridBuffer, offset, readBuf, 0, bytesPerGrid);
    this.device.queue.submit([enc.finish()]);

    await readBuf.mapAsync(GPUMapMode.READ);
    const arrU32 = new Uint32Array(readBuf.getMappedRange()).slice();
    readBuf.unmap();

    const arrU8 = new Uint8Array(cellsPerGrid);
    for (let i = 0; i < cellsPerGrid; i++) {
      arrU8[i] = arrU32[i] ? 1 : 0;
    }
    return arrU8;
  }
}

// ---------- GA utilities ----------
function createRandomSequence(steps: number, gridSize: number): Uint32Array {
  const seq = new Uint32Array(steps);
  for (let i = 0; i < steps; i++) {
    const r = Math.random();
    if (r < 0.3) {
      const pos = Math.floor(Math.random() * (gridSize * gridSize));
      const pattern = Math.floor(Math.random() * 16);
      seq[i] = (2 << 24) | (pattern << 16) | pos;
    } else {
      seq[i] = 0;
    }
  }
  return seq;
}

function crossover(parent1: Uint32Array, parent2: Uint32Array): Uint32Array {
  const child = new Uint32Array(parent1.length);
  const cp = Math.floor(Math.random() * parent1.length);
  child.set(parent1.slice(0, cp));
  child.set(parent2.slice(cp), cp);
  return child;
}

function mutate(seq: Uint32Array, rate: number, gridSize: number): void {
  for (let i = 0; i < seq.length; i++) {
    if (Math.random() < rate) {
      const r = Math.random();
      if (r < 0.3) {
        const pos = Math.floor(Math.random() * (gridSize * gridSize));
        const pattern = Math.floor(Math.random() * 16);
        seq[i] = (2 << 24) | (pattern << 16) | pos;
      } else {
        seq[i] = 0;
      }
    }
  }
}

// ---------- Drawing helpers ----------
function clearCanvas(ctx: CanvasRenderingContext2D, w: number, h: number) {
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, w, h);
}

function drawGrid(ctx: CanvasRenderingContext2D, grid: Uint8Array, size: number, offsetX: number, offsetY: number, cellSize: number) {
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const val = grid[y * size + x];
      ctx.fillStyle = val ? '#111' : '#f5f5f5';
      ctx.fillRect(offsetX + x * cellSize, offsetY + y * cellSize, cellSize - 1, cellSize - 1);
    }
  }
}

// fitness chart
function drawFitnessChart(canvas: HTMLCanvasElement, bestHistory: number[], avgHistory: number[]) {
  const ctx = canvas.getContext('2d')!;
  const w = canvas.width;
  const h = canvas.height;
  clearCanvas(ctx, w, h);

  const margin = 32;
  const plotW = w - margin - 8;
  const plotH = h - margin - 16;
  const ox = margin;
  const oy = 8;

  ctx.strokeStyle = '#ccc';
  ctx.beginPath();
  ctx.moveTo(ox, oy);
  ctx.lineTo(ox, oy + plotH);
  ctx.lineTo(ox + plotW, oy + plotH);
  ctx.stroke();

  const total = Math.max(1, Math.max(bestHistory.length, avgHistory.length));
  const candidateMax = Math.max(1, ...(bestHistory.length ? bestHistory : [1]), ...(avgHistory.length ? avgHistory : [1]));
  const maxFitness = candidateMax;
  const yAt = (v: number) => oy + plotH - (v / maxFitness) * plotH;
  const xAt = (i: number) => ox + (total <= 1 ? 0 : (i / (total - 1)) * plotW);

  if (avgHistory.length > 0) {
    ctx.beginPath();
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#4CAF50';
    for (let i = 0; i < avgHistory.length; i++) {
      const x = xAt(i), y = yAt(avgHistory[i]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  if (bestHistory.length > 0) {
    ctx.beginPath();
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#2196F3';
    for (let i = 0; i < bestHistory.length; i++) {
      const x = xAt(i), y = yAt(bestHistory[i]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  // legend
  ctx.fillStyle = '#2196F3';
  ctx.fillRect(ox + 6, oy + plotH + 4, 10, 6);
  ctx.fillStyle = '#222';
  ctx.fillText('Best', ox + 22, oy + plotH + 12);

  ctx.fillStyle = '#4CAF50';
  ctx.fillRect(ox + 72, oy + plotH + 4, 10, 6);
  ctx.fillStyle = '#222';
  ctx.fillText('Avg', ox + 88, oy + plotH + 12);
}

// ---------- CPU CA ----------
function conwayStep(grid: Uint8Array, size: number): Uint8Array {
  const out = new Uint8Array(size * size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      let count = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          const nx = (x + dx + size) % size;
          const ny = (y + dy + size) % size;
          count += grid[ny * size + nx];
        }
      }
      const curr = grid[y * size + x];
      let next = 0;
      if (curr === 1 && (count === 2 || count === 3)) next = 1;
      else if (curr === 0 && count === 3) next = 1;
      out[y * size + x] = next;
    }
  }
  return out;
}

class ManualSim {
  grid: Uint8Array;
  size: number;
  agentX: number;
  agentY: number;
  maxSteps: number;
  stepCount: number;
  constructor(size = GRID_SIZE, maxSteps = 10) {
    this.size = size;
    this.grid = new Uint8Array(size * size);
    this.agentX = Math.floor(size / 2);
    this.agentY = Math.floor(size / 2);
    this.maxSteps = maxSteps;
    this.stepCount = 0;
  }

  reset() {
    this.grid.fill(0);
    this.agentX = Math.floor(this.size / 2);
    this.agentY = Math.floor(this.size / 2);
    this.stepCount = 0;
  }

  step(action: number) {
    let writePattern: number[] | null = null;
    if (action >= 0 && action <= 3) {
      if (action === 0) this.agentY = (this.agentY - 1 + this.size) % this.size;
      if (action === 1) this.agentY = (this.agentY + 1) % this.size;
      if (action === 2) this.agentX = (this.agentX - 1 + this.size) % this.size;
      if (action === 3) this.agentX = (this.agentX + 1) % this.size;
    } else if (action === 4) {
      // noop
    } else if (action >= 5) {
      const idx = action - 5;
      writePattern = [
        (idx >> 3) & 1,
        (idx >> 2) & 1,
        (idx >> 1) & 1,
        idx & 1
      ];
    }

    // CA update then write
    this.grid = Uint8Array.from(conwayStep(this.grid, this.size));
    if (writePattern) {
      for (let ry = 0; ry < 2; ry++) {
        for (let rx = 0; rx < 2; rx++) {
          const y = (this.agentY + ry + this.size) % this.size;
          const x = (this.agentX + rx + this.size) % this.size;
          this.grid[y * this.size + x] = writePattern[ry * 2 + rx];
        }
      }
    }

    this.stepCount++;
    const done = this.stepCount >= this.maxSteps;
    return { grid: this.grid.slice(), done };
  }

  calculateFitness(target: Uint8Array | null): number {
    if (!target) return 0;
    let match = 0;
    for (let i = 0; i < this.grid.length; i++) if (this.grid[i] === target[i]) match++;
    return (match / this.grid.length) * 100;
  }
}

// ---------- App main ----------
async function mainApp() {
  const ui: UIElements = {
    mode: document.getElementById('mode') as HTMLSelectElement,
    population: document.getElementById('population') as HTMLInputElement,
    steps: document.getElementById('steps') as HTMLInputElement,
    generations: document.getElementById('generations') as HTMLInputElement,
    elite: document.getElementById('elite') as HTMLInputElement,
    mut: document.getElementById('mut') as HTMLInputElement,
    vizFreq: document.getElementById('vizFreq') as HTMLInputElement,
    startBtn: document.getElementById('startBtn') as HTMLButtonElement,
    stopBtn: document.getElementById('stopBtn') as HTMLButtonElement,
    canvas: document.getElementById('gridCanvas') as HTMLCanvasElement,
    fitnessCanvas: document.getElementById('fitnessCanvas') as HTMLCanvasElement,
    log: document.getElementById('log') as HTMLElement,
    patternCanvas: document.getElementById('patternCanvas') as HTMLCanvasElement,
    clearPatternBtn: document.getElementById('clearPatternBtn') as HTMLButtonElement,
    savePatternBtn: document.getElementById('savePatternBtn') as HTMLButtonElement,
    statsDiv: document.getElementById('stats') as HTMLElement,
    seqStrip: document.getElementById('seqStrip') as HTMLCanvasElement
  };

  // sizing
  const mainCellSize = Math.floor(200 / GRID_SIZE);
  ui.canvas.width = 10 + GRID_SIZE * mainCellSize * 2 + 40 + 200;
  ui.canvas.height = mainCellSize * GRID_SIZE + 90;
  ui.fitnessCanvas.width = 520;
  ui.fitnessCanvas.height = 160;

  const patternCellSize = Math.floor(300 / GRID_SIZE);
  ui.patternCanvas.width = patternCellSize * GRID_SIZE + 2;
  ui.patternCanvas.height = patternCellSize * GRID_SIZE + 2;

  // in-memory state
  let globalTargetPattern: Uint8Array = new Uint8Array(GRID_SIZE * GRID_SIZE);
  let bestSequence: Uint32Array | null = null;
  let bestFitness = 0;
  let totalSequencesEvaluated = 0;
  const uniqueSequences = new Set<string>();

  // redraw pattern canvas
  const pctx = ui.patternCanvas.getContext('2d')!;
  function redrawPatternCanvas() {
    pctx.fillStyle = '#f5f5f5';
    pctx.fillRect(0, 0, ui.patternCanvas.width, ui.patternCanvas.height);
    for (let y = 0; y < GRID_SIZE; y++) {
      for (let x = 0; x < GRID_SIZE; x++) {
        if (globalTargetPattern[y * GRID_SIZE + x]) {
          pctx.fillStyle = '#111';
          pctx.fillRect(x * patternCellSize, y * patternCellSize, patternCellSize - 1, patternCellSize - 1);
        }
      }
    }
    pctx.strokeStyle = '#ddd';
    for (let i = 0; i <= GRID_SIZE; i++) {
      pctx.beginPath();
      pctx.moveTo(i * patternCellSize, 0);
      pctx.lineTo(i * patternCellSize, ui.patternCanvas.height);
      pctx.stroke();
      pctx.beginPath();
      pctx.moveTo(0, i * patternCellSize);
      pctx.lineTo(ui.patternCanvas.width, i * patternCellSize);
      pctx.stroke();
    }
  }
  redrawPatternCanvas();

  ui.patternCanvas.addEventListener('click', (e) => {
    const rect = ui.patternCanvas.getBoundingClientRect();
    const x = Math.floor((e.clientX - rect.left) / patternCellSize);
    const y = Math.floor((e.clientY - rect.top) / patternCellSize);
    if (x >= 0 && x < GRID_SIZE && y >= 0 && y < GRID_SIZE) {
      const idx = y * GRID_SIZE + x;
      globalTargetPattern[idx] = 1 - globalTargetPattern[idx];
      redrawPatternCanvas();
      updateStatsPanel();
    }
  });

  ui.clearPatternBtn.addEventListener('click', () => {
    globalTargetPattern.fill(0);
    redrawPatternCanvas();
    updateStatsPanel();
    log(ui.log, 'Pattern cleared (memory).');
  });

  ui.savePatternBtn.addEventListener('click', () => {
    // Save target pattern in-memory only (no file)
    log(ui.log, 'Pattern saved to memory (current target pattern).');
    updateStatsPanel();
  });

  // stats panel updater
  function updateStatsPanel() {
    ui.statsDiv.innerHTML = `
      <b>Stats</b><br>
      Best Fitness: ${(bestFitness * 100).toFixed(2)}%<br>
      Total Sequences Evaluated: ${totalSequencesEvaluated}<br>
      Unique Sequences Seen: ${uniqueSequences.size}<br>
      Best Sequence Length: ${bestSequence ? bestSequence.length : 0}
    `;
    // redraw symbolic strip
    drawSequenceStrip();
  }

  // main canvas context
  const ctx = ui.canvas.getContext('2d')!;

  // fitness histories
  const bestHistory: number[] = [];
  const avgHistory: number[] = [];

  let stopFlag = false;

  ui.startBtn.addEventListener('click', async () => {
    ui.startBtn.disabled = true;
    ui.stopBtn.disabled = false;
    stopFlag = false;
    ui.log.textContent = '';
    try {
      const mode = ui.mode.value;
      if (mode === 'evolve') {
        await runEvolutionMode();
      } else if (mode === 'manual') {
        runManualMode();
      } else if (mode === 'demo') {
        await runDemoMode();
      } else {
        log(ui.log, `Unknown mode: ${mode}`);
      }
    } catch (err: any) {
      log(ui.log, 'Error:', err.message || String(err));
    } finally {
      ui.startBtn.disabled = false;
      ui.stopBtn.disabled = true;
    }
  });

  ui.stopBtn.addEventListener('click', () => {
    stopFlag = true;
    ui.stopBtn.disabled = true;
  });

  // ---------- EVOLVE ----------
  async function runEvolutionMode() {
    if (globalTargetPattern.reduce((a, b) => a + b, 0) === 0) {
      log(ui.log, 'Error: No target pattern set. Draw a pattern first.');
      return;
    }

    log(ui.log, 'Initializing WebGPU device...');
    const device = await requestDevice();
    log(ui.log, 'WebGPU device ready');

    const population = Math.max(8, parseInt(ui.population.value) || 128);
    const steps = Math.max(1, parseInt(ui.steps.value) || 10);
    const generations = Math.max(1, parseInt(ui.generations.value) || MAX_GENERATIONS_DEFAULT);
    const eliteFrac = Math.max(0.01, Math.min(0.5, parseFloat(ui.elite.value) || 0.2));
    const mutRate = Math.max(0.0, Math.min(1.0, parseFloat(ui.mut.value) || 0.1));
    const vizEvery = Math.max(1, parseInt(ui.vizFreq.value) || 10);

    const evolver = new GPUEvolver(device, GRID_SIZE, population);

    const targetU32 = new Uint32Array(GRID_SIZE * GRID_SIZE);
    for (let i = 0; i < targetU32.length; i++) targetU32[i] = globalTargetPattern[i] ? 1 : 0;
    evolver.setTarget(targetU32);

    log(ui.log, `Target set: ${targetU32.reduce((a, b) => a + b, 0)} cells alive`);
    log(ui.log, `Starting evolution: pop=${population}, steps=${steps}, gens=${generations}`);

    // init population
    const population_seqs: Uint32Array[] = [];
    for (let i = 0; i < population; i++) population_seqs.push(createRandomSequence(steps, GRID_SIZE));
    const eliteCount = Math.max(1, Math.floor(population * eliteFrac));

    // reset stats
    bestHistory.length = 0;
    avgHistory.length = 0;
    bestSequence = null;
    bestFitness = 0;
    totalSequencesEvaluated = 0;
    uniqueSequences.clear();
    updateStatsPanel();
    drawFitnessChart(ui.fitnessCanvas, bestHistory, avgHistory);

    for (let gen = 0; gen < generations; gen++) {
      if (stopFlag) {
        log(ui.log, 'Evolution stopped by user');
        break;
      }

      // pack actions
      const actionPack = new Uint32Array(population * steps);
      for (let i = 0; i < population; i++) actionPack.set(population_seqs[i], i * steps);

      // evaluate
      let fitness: Float32Array;
      try {
        fitness = await evolver.evaluate(actionPack, steps);
      } catch (e) {
        log(ui.log, 'GPU evaluation error; aborting:', String(e));
        break;
      }

      // update totals & unique
      totalSequencesEvaluated += population;
      for (let i = 0; i < population; i++) uniqueSequences.add(Array.from(population_seqs[i]).join(','));

      // stats
      let bestIdx = 0;
      let bestFit = fitness[0];
      let sumFit = 0;
      for (let i = 0; i < fitness.length; i++) {
        sumFit += fitness[i];
        if (fitness[i] > bestFit) {
          bestFit = fitness[i];
          bestIdx = i;
        }
      }
      const avgFit = sumFit / fitness.length;

      if (bestFit > bestFitness) {
        bestFitness = bestFit;
        bestSequence = new Uint32Array(population_seqs[bestIdx]);
      }

      bestHistory.push(bestFit);
      avgHistory.push(avgFit);

      // visualization
      if ((gen % vizEvery) === 0 || gen === generations - 1) {
        clearCanvas(ctx, ui.canvas.width, ui.canvas.height);
        const leftX = 10;
        const topY = 10;
        const cell = mainCellSize;
        const gap = 20;

        // target
        drawGrid(ctx, globalTargetPattern, GRID_SIZE, leftX, topY, cell);
        ctx.fillStyle = '#000';
        ctx.font = '12px sans-serif';
        ctx.fillText('Target', leftX, topY + cell * GRID_SIZE + 18);

        // best grid from GPU
        let bestGrid: Uint8Array = new Uint8Array(GRID_SIZE * GRID_SIZE);
        try {
          bestGrid = await evolver.getGrid(bestIdx);
        } catch (err) {
          // empty if not available
        }
        const rightX = leftX + cell * GRID_SIZE + gap;
        drawGrid(ctx, bestGrid, GRID_SIZE, rightX, topY, cell);
        ctx.fillStyle = '#000';
        ctx.fillText(`Best (Gen ${gen + 1}) ${(bestFit * 100).toFixed(2)}%`, rightX, topY + cell * GRID_SIZE + 18);

        drawFitnessChart(ui.fitnessCanvas, bestHistory, avgHistory);
      }

      log(ui.log, `Gen ${gen + 1}/${generations} | Best: ${(bestFit * 100).toFixed(2)}% | Avg: ${(avgFit * 100).toFixed(2)}%`);

      // breeding
      const indices = Array.from({ length: population }, (_, i) => i);
      indices.sort((a, b) => fitness[b] - fitness[a]);
      const elites = indices.slice(0, eliteCount).map(i => new Uint32Array(population_seqs[i]));
      const newPop: Uint32Array[] = [];
      for (const e of elites) newPop.push(new Uint32Array(e));
      while (newPop.length < population) {
        const p1 = elites[Math.floor(Math.random() * elites.length)];
        const p2 = elites[Math.floor(Math.random() * elites.length)];
        const child = crossover(p1, p2);
        mutate(child, mutRate, GRID_SIZE);
        newPop.push(child);
      }
      population_seqs.length = 0;
      population_seqs.push(...newPop.slice(0, population));

      updateStatsPanel();
    }

    log(ui.log, `Evolution complete. Best overall fitness: ${(bestFitness * 100).toFixed(2)}%`);
    updateStatsPanel();
  }

  // ---------- MANUAL ----------
  function runManualMode() {
    log(ui.log, 'Manual mode: Arrow keys/Space/0-F to write. S=save pattern to memory. C=clear. Q=quit.');

    const steps = Math.max(1, parseInt(ui.steps.value) || 10);
    const sim = new ManualSim(GRID_SIZE, steps);

    let active = true;

    function render() {
      clearCanvas(ctx, ui.canvas.width, ui.canvas.height);
      const leftX = 10;
      const topY = 10;
      const cell = mainCellSize;
      const gap = 20;

      drawGrid(ctx, globalTargetPattern, GRID_SIZE, leftX, topY, cell);
      ctx.fillStyle = '#000';
      ctx.font = '12px sans-serif';
      ctx.fillText('Target', leftX, topY + cell * GRID_SIZE + 18);

      const manualX = leftX + cell * GRID_SIZE + gap;
      drawGrid(ctx, sim.grid, GRID_SIZE, manualX, topY, cell);

      ctx.strokeStyle = 'cyan';
      ctx.lineWidth = 2;
      ctx.strokeRect(manualX + sim.agentX * cell - 1, topY + sim.agentY * cell - 1, cell * 2 - 2, cell * 2 - 2);

      ctx.fillStyle = '#000';
      ctx.fillText(`Manual Mode | Step ${sim.stepCount}/${sim.maxSteps}`, manualX, topY + cell * GRID_SIZE + 18);

      if (globalTargetPattern.reduce((a, b) => a + b, 0) > 0) {
        const fit = sim.calculateFitness(globalTargetPattern);
        ctx.fillText(`Fitness: ${fit.toFixed(2)}%`, manualX + 140, topY + cell * GRID_SIZE + 18);
      }
    }

    render();

    function handleKey(ev: KeyboardEvent) {
      if (!active) return;
      if (ev.key == null) return;
      const key = ev.key.toLowerCase();

      if (key === 'q' || key === 'escape') {
        log(ui.log, 'Manual mode exited.');
        window.removeEventListener('keydown', handleKey);
        active = false;
        return;
      }

      if (key === 's') {
        // Save current manual grid as the target pattern (in memory)
        globalTargetPattern = new Uint8Array(sim.grid);
        redrawPatternCanvas();
        updateStatsPanel();
        log(ui.log, 'Saved current manual grid as target pattern (memory).');
        return;
      }

      if (key === 'c') {
        sim.reset();
        render();
        return;
      }

      const hexKeys = '0123456789abcdef';
      let action: number | null = null;
      if (key === 'arrowup' || key === 'up') action = 0;
      else if (key === 'arrowdown' || key === 'down') action = 1;
      else if (key === 'arrowleft' || key === 'left') action = 2;
      else if (key === 'arrowright' || key === 'right') action = 3;
      else if (key === ' ' || key === 'spacebar' || key === 'space') action = 4;
      else if (hexKeys.indexOf(key) !== -1) action = 5 + hexKeys.indexOf(key);

      if (action === null) return;

      const res = sim.step(action);
      render();

      if (res.done) {
        // stop and reset as requested
        log(ui.log, 'Manual max steps reached — stopping and resetting manual mode.');
        window.removeEventListener('keydown', handleKey);
        active = false;
        // reset sim
        sim.reset();
        render();
      }
    }

    window.addEventListener('keydown', handleKey);
  }

  // ---------- DEMO ----------
  async function runDemoMode() {
    if (!bestSequence) {
      log(ui.log, 'No best sequence in memory. Run evolve first to produce one.');
      return;
    }
    log(ui.log, 'Starting demo playback of best sequence (in-memory).');
    await playDemoSequence(bestSequence);
  }

  // decode action packed integer
  function decodeAction(a: number) {
    if (!a) return { type: 'noop' } as const;
    const atype = (a >>> 24) & 0xff;
    const pat = (a >>> 16) & 0xff;
    const pos = a & 0xffff;
    if (atype === 2) {
      const wy = Math.floor(pos / GRID_SIZE);
      const wx = pos % GRID_SIZE;
      const bits = [
        (pat >> 3) & 1,
        (pat >> 2) & 1,
        (pat >> 1) & 1,
        pat & 1
      ];
      return { type: 'write' as const, wx, wy, patBits: bits };
    }
    return { type: 'noop' as const };
  }

  // Play sequence with CA semantics identical to the shader: CA step -> apply write to next-state
  async function playDemoSequence(sequence: Uint32Array) {
    const size = GRID_SIZE;
    let grid = new Uint8Array(size * size); // initial state empty
    const leftX = 10;
    const topY = 10;
    const cell = mainCellSize;
    const gap = 20;
    const manualX = leftX + cell * GRID_SIZE + gap;
    const frameInterval = 380;

    for (let f = 0; f < sequence.length + 20 && !stopFlag; f++) {
      if (f > 0 && f <= sequence.length) {
        const a = sequence[f - 1];
        const dec = decodeAction(a);
        // CA update first (produce next)
        let nextGrid = Uint8Array.from(conwayStep(grid, size));
        // apply write to nextGrid if write action
        if (dec.type === 'write' && dec.wx !== undefined && dec.wy !== undefined && Array.isArray(dec.patBits)) {
          for (let ry = 0; ry < 2; ry++) {
            for (let rx = 0; rx < 2; rx++) {
              const y = (dec.wy + ry + size) % size;
              const x = (dec.wx + rx + size) % size;
              nextGrid[y * size + x] = dec.patBits[ry * 2 + rx];
            }
          }
        }
        // now set grid = nextGrid for next iteration & display
        grid = nextGrid;
      }

      // render
      clearCanvas(ctx, ui.canvas.width, ui.canvas.height);
      // draw target
      drawGrid(ctx, globalTargetPattern, GRID_SIZE, leftX, topY, cell);
      ctx.fillStyle = '#000';
      ctx.font = '12px sans-serif';
      ctx.fillText('Target', leftX, topY + cell * GRID_SIZE + 18);

      // draw demo grid
      drawGrid(ctx, grid, GRID_SIZE, manualX, topY, cell);
      ctx.fillStyle = '#000';
      ctx.fillText(`Demo (${Math.min(f, sequence.length)}/${sequence.length})`, manualX, topY + cell * GRID_SIZE + 18);

      // highlight agent (if last action was a write)
      if (f > 0 && f <= sequence.length) {
        const a = sequence[f - 1];
        const dec = decodeAction(a);
        if (dec.type === 'write' && dec.wx !== undefined && dec.wy !== undefined) {
          ctx.strokeStyle = 'red';
          ctx.lineWidth = 2;
          ctx.strokeRect(manualX + dec.wx * cell - 1, topY + dec.wy * cell - 1, cell * 2 - 2, cell * 2 - 2);
        }
      }

      // show fitness if target exists
      if (globalTargetPattern.reduce((a, b) => a + b, 0) > 0) {
        let match = 0;
        for (let i = 0; i < grid.length; i++) if (grid[i] === globalTargetPattern[i]) match++;
        const fit = (match / grid.length) * 100;
        ctx.fillStyle = '#000';
        ctx.fillText(`Fitness: ${fit.toFixed(2)}%`, manualX + 140, topY + cell * GRID_SIZE + 18);
      }

      await new Promise(r => setTimeout(r, frameInterval));
    }

    log(ui.log, 'Demo finished.');
  }

  // ---------- Sequence strip (symbolic visualization) ----------
  function drawSequenceStrip() {
    const canvas = ui.seqStrip;
    const ctx = canvas.getContext('2d')!;
    clearCanvas(ctx, canvas.width, canvas.height);
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    if (!bestSequence) {
      ctx.fillStyle = '#666';
      ctx.font = '12px sans-serif';
      ctx.fillText('No best sequence yet. Run evolve to generate one.', 8, 20);
      return;
    }

    const maxShow = Math.min(bestSequence.length, Math.floor(canvas.width / 8));
    const itemW = Math.floor(canvas.width / maxShow);
    for (let i = 0; i < maxShow; i++) {
      const a = bestSequence[i];
      const dec = decodeAction(a);
      const x = i * itemW + 4;
      const y = 6;
      // write actions: dark square with pattern shading; noop: light square
      if (dec.type === 'write') {
        ctx.fillStyle = '#222';
        ctx.fillRect(x, y, itemW - 8, canvas.height - 12);
        // indicate pattern bits as little 2x2 inside
        const w = itemW - 12;
        const cell = Math.max(1, Math.floor(w / 2));
        for (let ry = 0; ry < 2; ry++) {
          for (let rx = 0; rx < 2; rx++) {
            const bit = dec.patBits[ry * 2 + rx];
            ctx.fillStyle = bit ? '#fff' : '#555';
            const sx = x + 2 + rx * cell;
            const sy = y + 2 + ry * cell;
            ctx.fillRect(sx, sy, cell - 1, cell - 1);
          }
        }
      } else {
        ctx.fillStyle = '#e6e6e6';
        ctx.fillRect(x, y, itemW - 8, canvas.height - 12);
      }
      // border for current
      ctx.strokeStyle = '#ccc';
      ctx.strokeRect(x, y, itemW - 8, canvas.height - 12);
    }
  }

  // keyboard helper: copy best sequence json to clipboard with 'D' (optional)
  window.addEventListener('keydown', (e) => {
    if (e.key.toLowerCase() === 'd') {
      if (!bestSequence) {
        log(ui.log, 'No best sequence to export.');
        return;
      }
      if (navigator.clipboard) {
        navigator.clipboard.writeText(JSON.stringify(Array.from(bestSequence))).then(() => {
          log(ui.log, 'Best sequence copied to clipboard (JSON).');
        }, () => {
          log(ui.log, 'Could not copy sequence to clipboard.');
        });
      } else {
        log(ui.log, 'Clipboard not available.');
      }
    }
  });

  // initial UI state
  updateStatsPanel();
  drawFitnessChart(ui.fitnessCanvas, bestHistory, avgHistory);
  drawSequenceStrip();

  log(ui.log, 'Ready. Draw a target pattern, choose a mode, and click Start.');
}

mainApp().catch((err) => {
  console.error(err);
  alert('App error: ' + (err as any).message);
});
