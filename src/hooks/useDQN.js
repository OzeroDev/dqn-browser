import { useCallback, useEffect, useRef, useState } from "react";
import * as tf from '@tensorflow/tfjs';

function createDQNModel(nObs, nActions) {
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 128, activation: 'relu', inputShape: [nObs] }));
  model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
  model.add(tf.layers.dense({ units: nActions }));
  return model;
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
  const [exploreCount, setExploreCount] = useState(0);
  const [exploitCount, setExploitCount] = useState(0);

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

  // we'll build JS arrays for batches and convert to tensors when needed
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
    policyRef.current = createDQNModel(nObs, nActions);
    // smaller LR to reduce instability
    optimizerRef.current = tf.train.adam(5e-4);

    // create a target network for stable Q-learning
    // target network starts with same weights as policy
    const target = createDQNModel(nObs, nActions);
    target.setWeights(policyRef.current.getWeights());
    policyRef.current.target = target;
    // step counter for periodic target updates
    policyRef.current._targetUpdateCounter = 0;

    // warm up with a single prediction to ensure weights are created
    const zeroRow = Array(nObs).fill(0);
    const zeroBatch = Array.from({ length: BATCH_SIZE }, () => [...zeroRow]);
    tf.tidy(() => {
      const t = tf.tensor2d(zeroBatch);
      policyRef.current.predict(t);
    });
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

  // returns { action, greedy }
  const selectAction = (stateArr, eps) => {
    const features = createFeatures(stateArr);
    // predict using tfjs
    const qData = tf.tidy(() => {
      const input = tf.tensor2d([features]);
      const out = policyRef.current.predict(input);
      const arr = out.arraySync();
      return arr[0];
    });

    let bestIdx = 0;
    for (let i = 1; i < qData.length; i++) {
      if (qData[i] > qData[bestIdx]) bestIdx = i;
    }

    const greedy = Math.random() > eps;
    if (greedy) return { action: bestIdx, greedy: true };
    return { action: Math.floor(Math.random() * nActions), greedy: false };
  };

  // Return Q-values for a single raw grid position [r, c]
  const getQValues = useCallback((rawPos) => {
    if (!policyRef.current) return null;
    const features = createFeatures(rawPos);
    try {
      const qData = tf.tidy(() => {
        const input = tf.tensor2d([features]);
        const out = policyRef.current.predict(input);
        return out.arraySync()[0];
      });
      return qData.slice();
    } catch (e) {
      return null;
    }
  }, []);

  const optimizeModel = () => {
    if (memoryRef.current.length < BATCH_SIZE) return;

    const batch = memoryRef.current.sample(BATCH_SIZE);

    const statesBatch = [];
    const nextStatesBatch = [];
    for (let i = 0; i < BATCH_SIZE; i++) {
      statesBatch.push(batch[i].state);
      nextStatesBatch.push(batch[i].nextState);
    }

    // Use tfjs to compute targets and apply gradients
    // Capture and log loss occasionally for debugging
    tf.tidy(() => {
      const statesTensor = tf.tensor2d(statesBatch);
      const nextStatesTensor = tf.tensor2d(nextStatesBatch);

      const qPred = policyRef.current.predict(statesTensor);
  // use target network for next-Q predictions if available
  const targetNet = policyRef.current.target || policyRef.current;
  const nextQPred = targetNet.predict(nextStatesTensor);

      const qPredArr = qPred.arraySync();
      const nextQArr = nextQPred.arraySync();
      const nextQMax = nextQArr.map((row) => Math.max(...row));

      const targetData = qPredArr.map((row, i) => {
        const tdTarget = batch[i].reward + (batch[i].done ? 0 : GAMMA * nextQMax[i]);
        const updated = row.slice();
        updated[batch[i].action] = tdTarget;
        return updated;
      });

      const targetTensor = tf.tensor2d(targetData);

      // compute gradients manually so we can clip them
      const varGrads = tf.variableGrads(() => {
        const preds = policyRef.current.predict(statesTensor);
        const loss = tf.losses.meanSquaredError(targetTensor, preds).mean();
        return loss;
      });

      try {
        // clip gradients to avoid explosions
        const CLIP_VAL = 5.0;
        const clippedGrads = {};
        Object.keys(varGrads.grads).forEach((k) => {
          clippedGrads[k] = tf.clipByValue(varGrads.grads[k], -CLIP_VAL, CLIP_VAL);
        });

        optimizerRef.current.applyGradients(clippedGrads);

        // occasional logging of loss
        if (Math.random() < 0.05) {
          const val = varGrads.value.dataSync()[0];
          // eslint-disable-next-line no-console
          console.debug('[DQN] optimize loss:', val.toFixed(6));
        }
      } finally {
        // dispose gradient tensors
        Object.values(varGrads.grads).forEach((t) => t.dispose());
        if (varGrads.value) varGrads.value.dispose();
      }

      // periodically update target network weights from policy
      policyRef.current._targetUpdateCounter += 1;
      const TARGET_UPDATE_EVERY = 200;
      if (policyRef.current._targetUpdateCounter % TARGET_UPDATE_EVERY === 0) {
        const weights = policyRef.current.getWeights();
        // setWeights copies values into target network
        if (policyRef.current.target) policyRef.current.target.setWeights(weights);
      }
    });
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
        // reset per-episode explore/exploit counts
        setExploreCount(0);
        setExploitCount(0);

        let rawState = envReset();
        let done = false;
        let episodeReward = 0;
        
        while (!done) {
          if (stopRef.current) break;

          const eps = computeEpsilon(globalStep);
          const { action, greedy } = selectAction(rawState, eps);

          // track explore vs exploit
          if (greedy) {
            setExploitCount((c) => c + 1);
          } else {
            setExploreCount((c) => c + 1);
          }

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
        // log episode reward for debugging
        // eslint-disable-next-line no-console
        console.debug('[DQN] episode', ep + 1, 'reward:', episodeReward.toFixed(3), 'avg10:', avg.toFixed(3));
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
    exploreCount,
    exploitCount,
    getQValues,
  };
}