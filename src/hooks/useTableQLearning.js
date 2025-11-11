import { useCallback, useEffect, useRef, useState } from "react";

function makeQTable(gridSize, nActions = 4) {
  const table = Array.from({ length: gridSize }, () =>
    Array.from({ length: gridSize }, () => Array.from({ length: nActions }, () => 0))
  );
  return table;
}

export default function useTableQLearning({ gridState, envStep, envReset } = {}) {
  const [training, setTraining] = useState(false);
  const [episode, setEpisode] = useState(0);
  const [totalSteps, setTotalSteps] = useState(0);
  const [lastReward, setLastReward] = useState(0);
  const [avgReward, setAvgReward] = useState(0);
  const [epsilon, setEpsilon] = useState(1.0);
  const [exploreCount, setExploreCount] = useState(0);
  const [exploitCount, setExploitCount] = useState(0);

  const qRef = useRef(null);
  const stopRef = useRef(false);
  const rewardHistoryRef = useRef([]);

  // hyperparams (can be parameterized later)
  const ALPHA = 0.5;
  const GAMMA = 0.99;
  const EPS_START = 1.0;
  const EPS_END = 0.05;
  const EPS_DECAY = 10000; // decay steps

  // initialize q-table when gridState changes
  useEffect(() => {
    const gs = gridState?.gridSize || 6;
    qRef.current = makeQTable(gs, 4);
  }, [gridState]);

  const isCellBlocked = useCallback(
    (r, c) => {
      if (!gridState) return false;
      if (gridState.blocks?.some((b) => b[0] === r && b[1] === c)) return true;
      const isPit = gridState.pit && gridState.pit[0] === r && gridState.pit[1] === c;
      const isGoal = gridState.goalPos && gridState.goalPos[0] === r && gridState.goalPos[1] === c;
      return isPit || isGoal;
    },
    [gridState]
  );

  const getQValues = useCallback(
    (rawPos) => {
      const [r, c] = rawPos;
      if (!qRef.current) return null;
      if (isCellBlocked(r, c)) return null;
      return qRef.current[r][c].slice();
    },
    [isCellBlocked]
  );

  const computeEpsilon = (t) =>
    EPS_END + (EPS_START - EPS_END) * Math.exp(-1.0 * t / EPS_DECAY);

  const selectAction = (stateArr, eps) => {
    const [r, c] = stateArr;
    const qs = qRef.current[r][c];
    // choose greedy action (maxQ) with probability 1 - eps
    const greedy = Math.random() > eps;
    if (greedy) {
      let bestIdx = 0;
      for (let i = 1; i < qs.length; i++) if (qs[i] > qs[bestIdx]) bestIdx = i;
      return { action: bestIdx, greedy: true };
    }
    return { action: Math.floor(Math.random() * qs.length), greedy: false };
  };

  const updateQ = (s, a, r, ns, done) => {
    const [sr, sc] = s;
    const [nr, nc] = ns;
    const q = qRef.current[sr][sc];
    const cur = q[a];
    const nextMax = done ? 0 : Math.max(...qRef.current[nr][nc]);
    const target = r + GAMMA * nextMax;
    qRef.current[sr][sc][a] = cur + ALPHA * (target - cur);
  };

  const startTraining = useCallback(
    async ({ episodes = 200 } = {}) => {
      if (!qRef.current) return;
      stopRef.current = false;
      setTraining(true);
      rewardHistoryRef.current = [];

      let globalStep = 0;
      for (let ep = 0; ep < episodes; ep++) {
        if (stopRef.current) break;

        setExploreCount(0);
        setExploitCount(0);

        let rawState = envReset();
        let done = false;
        let episodeReward = 0;

        while (!done) {
          if (stopRef.current) break;

          const eps = computeEpsilon(globalStep);
          const { action, greedy } = selectAction(rawState, eps);

          if (greedy) setExploitCount((c) => c + 1);
          else setExploreCount((c) => c + 1);

          const { obs, reward, terminated, truncated } = envStep(action);
          const nextRawState = obs;

          episodeReward += reward;

          updateQ(rawState, action, reward, nextRawState, terminated || truncated);

          rawState = nextRawState;
          globalStep += 1;

          if (globalStep % 10 === 0) {
            setTotalSteps(globalStep);
            setEpsilon(eps);
          }

          // small yield to keep UI responsive
          await new Promise((r) => setTimeout(r, 0));

          done = terminated || truncated;
        }

        setLastReward(episodeReward);
        rewardHistoryRef.current.push(episodeReward);
        if (rewardHistoryRef.current.length > 10) rewardHistoryRef.current.shift();
        const avg = rewardHistoryRef.current.reduce((a, b) => a + b, 0) / rewardHistoryRef.current.length;
        setAvgReward(avg);
        setEpisode(ep + 1);
      }

      setTraining(false);
    },
    [envReset, envStep]
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
