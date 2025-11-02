import { useRef, useState, useCallback } from "react";

class GridWorld {
  constructor(gridSize = 6, cellSize = 64, pitDeathReward = -1.0) {
    if (gridSize < 6) throw new Error("grid_size must be >= 6.");
    this.gridSize = gridSize;
    this.cellSize = cellSize;
    this.pitDeathReward = pitDeathReward;
    this.goalPos = [gridSize - 1, gridSize - 1];
    this.maxSteps = gridSize * gridSize * 2;
    this.steps = 0;
    this.blocks = new Set(["1,2", "2,2", "4,2"]);
    this.pit = [3, 3];
    this.agentPos = [0, 0];
  }

  reset() {
    this.agentPos = [0, 0];
    this.steps = 0;
    return [...this.agentPos];
  }

  step(action) {
    let [pr, pc] = this.agentPos;
    if (action === 0 && pr > 0) pr -= 1;
    else if (action === 1 && pr < this.gridSize - 1) pr += 1;
    else if (action === 2 && pc > 0) pc -= 1;
    else if (action === 3 && pc < this.gridSize - 1) pc += 1;

    if (!this.blocks.has(`${pr},${pc}`)) {
      this.agentPos = [pr, pc];
    }

    this.steps += 1;

    const onPit = pr === this.pit[0] && pc === this.pit[1];
    if (onPit) {
      const obs = [...this.agentPos];
      return { obs, reward: this.pitDeathReward, terminated: true, truncated: false };
    }

    const terminated = pr === this.goalPos[0] && pc === this.goalPos[1];
    const truncated = this.steps >= this.maxSteps;
    const reward = terminated ? 1.0 : -0.01;

    const obs = [...this.agentPos];
    return { obs, reward, terminated, truncated };
  }

  snapshot() {
    return {
      gridSize: this.gridSize,
      cellSize: this.cellSize,
      agentPos: [...this.agentPos],
      blocks: Array.from(this.blocks).map((s) => s.split(",").map((v) => Number(v))),
      pit: [...this.pit],
      goalPos: [...this.goalPos],
      steps: this.steps,
      maxSteps: this.maxSteps,
    };
  }
}

export default function useGridWorld({ gridSize = 6, cellSize = 72, pitDeathReward = -1.0 } = {}) {
  const envRef = useRef(new GridWorld(gridSize, cellSize, pitDeathReward));
  const [state, setState] = useState(envRef.current.snapshot());

  const reset = useCallback(() => {
    envRef.current.reset();
    setState(envRef.current.snapshot());
    return envRef.current.agentPos;
  }, []);

  const step = useCallback((action) => {
    const res = envRef.current.step(action);
    setState(envRef.current.snapshot());
    return res;
  }, []);

  return { state, reset, step };
}