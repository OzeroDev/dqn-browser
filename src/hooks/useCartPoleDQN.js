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

// CartPole physics (mirrors Gym's environment behavior)
function createCartPoleSim() {
  const gravity = 9.8;
  const massCart = 1.0;
  const massPole = 0.1;
  const totalMass = massCart + massPole;
  const length = 0.5; // actually half the pole's length
  const poleMassLength = massPole * length;
  const forceMag = 10.0;
  const tau = 0.02; // seconds between state updates

  const xThreshold = 2.4;
  const thetaThresholdRadians = (35 * Math.PI) / 180;

  let state = [0, 0, 0, 0]; // [x, xDot, theta, thetaDot]
  let steps = 0;

  function reset() {
    // small random init
    state = [
      (Math.random() - 0.5) * 0.1,
      (Math.random() - 0.5) * 0.1,
      (Math.random() - 0.5) * 0.1,
      (Math.random() - 0.5) * 0.1,
    ];
    steps = 0;
    return state.slice();
  }

  function step(action) {
    const x = state[0];
    const xDot = state[1];
    const theta = state[2];
    const thetaDot = state[3];

    const force = action === 1 ? forceMag : -forceMag;
    const cosTheta = Math.cos(theta);
    const sinTheta = Math.sin(theta);

    const temp = (force + poleMassLength * thetaDot * thetaDot * sinTheta) / totalMass;
    const thetaAcc =
      (gravity * sinTheta - cosTheta * temp) /
      (length * (4.0 / 3.0 - (massPole * cosTheta * cosTheta) / totalMass));
    const xAcc = temp - (poleMassLength * thetaAcc * cosTheta) / totalMass;

    // Integrate
    const newX = x + tau * xDot;
    const newXDot = xDot + tau * xAcc;
    const newTheta = theta + tau * thetaDot;
    const newThetaDot = thetaDot + tau * thetaAcc;

    state = [newX, newXDot, newTheta, newThetaDot];
    steps += 1;

    const terminated =
      newX < -xThreshold ||
      newX > xThreshold ||
      newTheta < -thetaThresholdRadians ||
      newTheta > thetaThresholdRadians;

    const truncated = steps >= 500;
    const reward = terminated ? 0.0 : 1.0;

    return {
      obs: state.slice(),
      reward,
      terminated,
      truncated,
    };
  }

  return { reset, step, xThreshold, thetaThresholdRadians };
}

export default function useCartPoleDQN() {
  const [training, setTraining] = useState(false);
  const [episode, setEpisode] = useState(0);
  const [totalSteps, setTotalSteps] = useState(0);
  const [lastReward, setLastReward] = useState(0);
  const [epsilon, setEpsilon] = useState(1.0);
  const [avgReward, setAvgReward] = useState(0);
  const [obs, setObs] = useState([0, 0, 0, 0]);

  const policyRef = useRef(null);
  const optimizerRef = useRef(null);
  const memoryRef = useRef(new ReplayMemory(10000));
  const stopRef = useRef(false);
  const rewardHistoryRef = useRef([]);

  const nActions = 2;
  const nObs = 4;

  const GAMMA = 0.99;
  const BATCH_SIZE = 64;
  const EPS_START = 0.9;
  const EPS_END = 0.01;
  const EPS_DECAY = 2500;

  const inferBatchGPURef = useRef(null);
  const trainStatesGPURef = useRef(null);
  const trainNextStatesGPURef = useRef(null);

  const simRef = useRef(createCartPoleSim());

  useEffect(() => {
    policyRef.current = createDQNModel(nObs, nActions);
    optimizerRef.current = tf.train.adam(3e-4);

    // warm up weights
    const zeroRow = Array(nObs).fill(0);
    const zeroBatch = Array.from({ length: BATCH_SIZE }, () => [...zeroRow]);
    tf.tidy(() => {
      const t = tf.tensor2d(zeroBatch);
      policyRef.current.predict(t);
    });
  }, []);

  const computeEpsilon = (t) =>
    EPS_END + (EPS_START - EPS_END) * Math.exp(-1.0 * t / EPS_DECAY);

  const selectAction = (stateArr, eps) => {
    const qData = tf.tidy(() => {
      const input = tf.tensor2d([stateArr]);
      const out = policyRef.current.predict(input);
      return out.arraySync()[0];
    });
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
    const statesBatch = [];
    const nextStatesBatch = [];
    for (let i = 0; i < BATCH_SIZE; i++) {
      statesBatch.push(batch[i].state);
      nextStatesBatch.push(batch[i].nextState);
    }

    tf.tidy(() => {
      const statesTensor = tf.tensor2d(statesBatch);
      const nextStatesTensor = tf.tensor2d(nextStatesBatch);

      const qPred = policyRef.current.predict(statesTensor);
      const nextQPred = policyRef.current.predict(nextStatesTensor);

      const qPredArr = qPred.arraySync();
      const nextQArr = nextQPred.arraySync();
      const nextQMax = nextQArr.map((row) => Math.max(...row));

      const targetData = qPredArr.map((row, i) => {
        const tdTarget = batch[i].reward + (batch[i].done ? 0 : GAMMA * nextQMax[i]);
        const updated = row.slice();
        updated[batch[i].action] = tdTarget;
        return updated;
      });

      const target = tf.tensor2d(targetData);

      optimizerRef.current.minimize(() => {
        const preds = policyRef.current.predict(statesTensor);
        const loss = tf.losses.meanSquaredError(target, preds).mean();
        return loss;
      }, true);
    });
  };

  const resetEnv = useCallback(() => {
    const s = simRef.current.reset();
    setObs(s.slice());
    return s.slice();
  }, []);

  const startTraining = useCallback(
    async ({ episodes = 50 } = {}) => {
      if (!policyRef.current) return;
      stopRef.current = false;
      setTraining(true);
      rewardHistoryRef.current = [];

      let globalStep = 0;
      for (let ep = 0; ep < episodes; ep++) {
        if (stopRef.current) break;

        let state = simRef.current.reset();
        setObs(state.slice());
        let done = false;
        let episodeReward = 0;

        while (!done) {
          if (stopRef.current) break;

          const eps = computeEpsilon(globalStep);
          const action = selectAction(state, eps);

          const { obs: nextState, reward, terminated, truncated } = simRef.current.step(action);
          episodeReward += reward;

          memoryRef.current.push({
            state: state.slice(),
            action,
            nextState: nextState.slice(),
            reward,
            done: terminated || truncated,
          });

          optimizeModel();

          state = nextState.slice();
          setObs(state.slice());
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
    []
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
    obs,
    resetEnv,
  };
}