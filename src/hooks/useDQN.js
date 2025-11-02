import { useCallback, useEffect, useRef, useState } from "react";

const { torch } = require("js-pytorch");

class DQN extends torch.nn.Module {
  constructor(nObs, nActions, device = "gpu") {
    super();
    this.l1 = new torch.nn.Linear(nObs, 128, device);
    this.r1 = new torch.nn.ReLU();
    this.l2 = new torch.nn.Linear(128, 128, device);
    this.r2 = new torch.nn.ReLU();
    this.l3 = new torch.nn.Linear(128, nActions, device);
  }
  forward(x) {
    let z = this.l1.forward(x);
    z = this.r1.forward(z);
    z = this.l2.forward(z);
    z = this.r2.forward(z);
    z = this.l3.forward(z);
    return z;
  }
}

class ReplayMemory {
  constructor(capacity) {
    this.capacity = capacity;
    this.buffer = [];
  }
  push(t) {
    if (this.buffer.length >= this.capacity) this.buffer.shift();
    this.buffer.push(t);
  }
  sample(k) {
    const res = [];
    for (let i = 0; i < k; i++) {
      res.push(this.buffer[Math.floor(Math.random() * this.buffer.length)]);
    }
    return res;
  }
  get length() {
    return this.buffer.length;
  }
}

export default function useDQN({ gridState, envStep, envReset }) {
  const [training, setTraining] = useState(false);
  const [episode, setEpisode] = useState(0);
  const [totalSteps, setTotalSteps] = useState(0);
  const [lastReward, setLastReward] = useState(0);
  const [epsilon, setEpsilon] = useState(1.0);
  const [avgReward, setAvgReward] = useState(0);

  const policyRef = useRef(null);
  const optimizerRef = useRef(null);
  const memoryRef = useRef(new ReplayMemory(10000));
  const stopRef = useRef(false);
  const rewardHistoryRef = useRef([]);

  const nActions = 4;
  const GAMMA = 0.99;
  const BATCH_SIZE = 64;
  const EPS_START = 0.9;
  const EPS_END = 0.05;
  const EPS_DECAY = 1000;

  // Calculate feature size based on grid
  const gridSize = gridState?.gridSize || 6;
  const nObs = 6; // [norm_row, norm_col, goal_row_dist, goal_col_dist, pit_row_dist, pit_col_dist]

  const inferBatchGPURef = useRef(null);
  const trainStatesGPURef = useRef(null);
  const trainNextStatesGPURef = useRef(null);

  // Store environment info for feature creation
  const envInfoRef = useRef({
    gridSize: 6,
    goalPos: [5, 5],
    pit: [3, 3],
    blocks: new Set(["1,2", "2,2", "4,2"])
  });

  useEffect(() => {
    if (gridState) {
      envInfoRef.current = {
        gridSize: gridState.gridSize,
        goalPos: gridState.goalPos,
        pit: gridState.pit,
        blocks: new Set(gridState.blocks?.map(b => `${b[0]},${b[1]}`))
      };
    }
  }, [gridState]);

  useEffect(() => {
    policyRef.current = new DQN(nObs, nActions, "gpu");
    optimizerRef.current = new torch.optim.Adam(policyRef.current.parameters(), 1e-3, 0);

    const zeroRow = Array(nObs).fill(0);
    const zeroBatch = Array.from({ length: BATCH_SIZE }, () => [...zeroRow]);

    inferBatchGPURef.current = torch.tensor(zeroBatch, false, "gpu");
    trainStatesGPURef.current = torch.tensor(zeroBatch, false, "gpu");
    trainNextStatesGPURef.current = torch.tensor(zeroBatch, false, "gpu");

    policyRef.current.forward(inferBatchGPURef.current);
  }, [nObs]);

  // Create rich feature representation from raw position
  const createFeatures = (rawPos) => {
    const [r, c] = rawPos;
    const { gridSize, goalPos, pit } = envInfoRef.current;
    
    const normR = r / (gridSize - 1);
    const normC = c / (gridSize - 1);
    
    // Relative distances to goal (normalized)
    const goalDistR = (goalPos[0] - r) / gridSize;
    const goalDistC = (goalPos[1] - c) / gridSize;
    
    // Manhattan distance to pit (normalized)
    const pitDistR = Math.abs(pit[0] - r) / gridSize;
    const pitDistC = Math.abs(pit[1] - c) / gridSize;
    
    return [
      normR,
      normC,
      goalDistR,
      goalDistC,
      pitDistR,
      pitDistC
    ];
  };

  const computeEpsilon = (t) =>
    EPS_END + (EPS_START - EPS_END) * Math.exp(-1.0 * t / EPS_DECAY);

  const selectAction = (stateArr, eps) => {
    const features = createFeatures(stateArr);
    const batch = inferBatchGPURef.current;
    
    for (let i = 0; i < nObs; i++) {
      batch.data[0][i] = Number(features[i]);
    }

    const q = policyRef.current.forward(batch);
    const qData = q.data[0];
    let bestIdx = 0;
    for (let i = 1; i < qData.length; i++) {
      if (qData[i] > qData[bestIdx]) bestIdx = i;
    }

    if (Math.random() > eps) return bestIdx;
    return Math.floor(Math.random() * nActions);
  };

  const optimizeModel = () => {
    if (memoryRef.current.length < BATCH_SIZE) return;

    const batch = memoryRef.current.sample(BATCH_SIZE);

    const statesGPU = trainStatesGPURef.current;
    const nextStatesGPU = trainNextStatesGPURef.current;
    
    for (let i = 0; i < BATCH_SIZE; i++) {
      const s = batch[i].state;
      const ns = batch[i].nextState;
      for (let j = 0; j < nObs; j++) {
        statesGPU.data[i][j] = s[j];
        nextStatesGPU.data[i][j] = ns[j];
      }
    }

    const q = policyRef.current.forward(statesGPU);
    const nextQ = policyRef.current.forward(nextStatesGPU);
    const nextQMax = nextQ.data.map((row) => Math.max(...row));

    const targetData = q.data.map((row, i) => {
      const tdTarget = batch[i].reward + (batch[i].done ? 0 : GAMMA * nextQMax[i]);
      const updated = row.slice();
      updated[batch[i].action] = tdTarget;
      return updated;
    });

    const target = torch.tensor(targetData, false, "gpu");

    const diff = q.sub(target);
    const loss = torch.mean(diff.mul(diff));
    loss.backward();
    optimizerRef.current.step();
    optimizerRef.current.zero_grad();
  };

  const startTraining = useCallback(
    async ({ episodes = 50 } = {}) => {
      if (!policyRef.current) return;
      stopRef.current = false;
      setTraining(true);
      rewardHistoryRef.current = [];

      let globalStep = 0;
      for (let ep = 0; ep < episodes; ep++) {
        if (stopRef.current) break;
        
        let rawState = envReset();
        let done = false;
        let episodeReward = 0;

        while (!done) {
          if (stopRef.current) break;

          const eps = computeEpsilon(globalStep);
          const action = selectAction(rawState, eps);

          const { obs, reward, terminated, truncated } = envStep(action);
          const nextRawState = obs;
          
          episodeReward += reward;

          // Store feature representations in replay
          const stateFeatures = createFeatures(rawState);
          const nextStateFeatures = createFeatures(nextRawState);

          memoryRef.current.push({
            state: stateFeatures,
            action,
            nextState: nextStateFeatures,
            reward,
            done: terminated || truncated,
          });

          optimizeModel();

          rawState = nextRawState;
          globalStep += 1;

          // throttle UI-only metrics
          if (globalStep % 10 === 0) {
            setTotalSteps(globalStep);
            setEpsilon(eps);
          }

          await new Promise((r) => setTimeout(r, 0));
          done = terminated || truncated;
        }

        // update lastReward with the last episode's total
        setLastReward(episodeReward);

        rewardHistoryRef.current.push(episodeReward);
        if (rewardHistoryRef.current.length > 10) {
          rewardHistoryRef.current.shift();
        }
        const avg = rewardHistoryRef.current.reduce((a, b) => a + b, 0) / rewardHistoryRef.current.length;
        setAvgReward(avg);
        setEpisode(ep + 1);
      }

      setTraining(false);
    },
    [envStep, envReset]
  );

  const stopTraining = useCallback(() => {
    stopRef.current = true;
  }, []);

  return {
    startTraining,
    stopTraining,
    training,
    episode,
    totalSteps,
    lastReward,
    epsilon,
    avgReward,
  };
}