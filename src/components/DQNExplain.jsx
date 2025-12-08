import React from "react";
import { renderToString } from "katex";
import "katex/dist/katex.min.css";

const rlBasics = [
  { title: "Agent", body: "Decides what to do; its behavior is the policy we improve." },
  { title: "Environment", body: "Supplies the next state and reward after each action." },
  { title: "State", body: "What the agent can observe right now (grid cell, velocity, pixels)." },
  { title: "Action", body: "A move the agent can take from this state (up, down, left, right)." },
  { title: "Reward", body: "Feedback: + pushes behavior forward, - discourages it." },
  { title: "Episode", body: "One run that ends at a goal or failure; learning spans many episodes." },
];

const qBasics = [
  {
    title: "What is Q?",
    body: "Q(s,a) estimates how good action a is in state s when you follow the current policy afterward.",
  },
  {
    title: "Table lookup",
    body: "Classic Q-learning stores one number per (state, action). Works great when the state space is small and discrete.",
  },
  {
    title: "Bellman target",
    body: "Each update aims for reward now + discounted best future reward: r + gamma * max_a' Q(s', a'). That single line blends immediate payoff with the best path forward.",
  },
  {
    title: "Exploration",
    body: "Epsilon-greedy: most of the time take the best known action, sometimes explore to discover better ones.",
  },
];

const dqnReasons = [
  {
    title: "Generalization",
    text: "Neural nets approximate Q so we do not need a giant table; similar states share statistical strength.",
  },
  {
    title: "Experience replay",
    text: "Store transitions in a buffer, sample them randomly to break correlations and reuse data efficiently.",
  },
  {
    title: "Target network",
    text: "A slower-moving copy of the online net provides stable targets and reduces feedback loops.",
  },
  {
    title: "Exploration schedule",
    text: "Start very exploratory (high epsilon), then decay to exploit what the model has learned.",
  },
];

const dqnLoop = [
  { title: "Collect", detail: "Step, observe (s,a,r,s',done), and push it into the replay buffer." },
  { title: "Sample", detail: "Draw a mini-batch uniformly (or with priorities) to decorrelate updates." },
  { title: "Target", detail: "Compute y = r + gamma * max_a' Q_target(s', a') when not done; otherwise y = r." },
  { title: "Update", detail: "Predict Q_online(s,a), minimize (y - Q_online)^2, backprop to update weights." },
  { title: "Sync", detail: "Every N steps, copy online weights to the target network (soft or hard update)." },
  { title: "Explore", detail: "Epsilon-greedy or noise-based exploration keeps discovering better actions." },
];

export default function DQNExplain() {
  return (
    <section className="bg-slate-900 text-slate-100 border-t border-slate-800">
      <div className="max-w-6xl mx-auto px-6 py-12">
        <header className="flex flex-col gap-2 mb-8">
          <h2 className="text-3xl font-bold">How Deep Q-Networks Work</h2>
          <p className="text-slate-300">
            A ground-up explanation of Deep Q-Networks (DQN): starting from reinforcement learning basics,
            building through tabular Q-learning, and culminating in the key innovations that make DQN effective
            for high-dimensional tasks.
          </p>
        </header>
        <div className="space-y-10">
          <section className="p-5 rounded-lg bg-slate-800/70 border border-slate-700">
            <h3 className="text-2xl font-semibold mb-4">Reinforcement Learning Basics</h3>
            <p className="text-slate-300 mb-4">
              The story starts with a loop: the agent observes, acts, and receives a reward from the environment.
              By chasing returns across many episodes, it slowly shapes a policy that favors actions leading to
              higher long-term gain. These ingredients set the stage for how we will score and improve actions later.
            </p>
            <div className="grid md:grid-cols-3 gap-3">
              {rlBasics.map((item) => (
                <div key={item.title} className="p-3 rounded bg-slate-900/70 border border-slate-800">
                  <div className="font-semibold text-slate-50">{item.title}</div>
                  <div className="text-sm text-slate-300">{item.body}</div>
                </div>
              ))}
            </div>
          </section>

          <section className="p-5 rounded-lg bg-slate-800/70 border border-slate-700">
            <h3 className="text-2xl font-semibold mb-4">Classic Tabular Q-Learning</h3>
            <p className="text-slate-300 mb-4">
              Next, we give the agent a ledger: a Q-table that stores how promising each action is in each state.
              Using the Bellman equation, the table entries are nudged toward “reward now plus discounted best future
              reward.” Exploration keeps the table from getting stuck on early guesses, while the update rule steadily
              improves action values.
            </p>
            <p className="text-slate-300 mb-4">
              Intuition for the Bellman target: treat each step as a mini investment. You bank the immediate reward r,
              then add the best future value you can reach, shrunk by gamma because tomorrow is slightly less certain
              than today. Repeating this backup across states propagates value estimates backward from goals to starts.
            </p>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="space-y-3">
                {qBasics.map((item) => (
                  <div key={item.title} className="p-3 rounded bg-slate-900/70 border border-slate-800">
                    <div className="font-semibold text-slate-50 mb-1">{item.title}</div>
                    <div className="text-sm text-slate-200">{item.body}</div>
                  </div>
                ))}
              </div>
              <div className="space-y-4">
                <EquationCard
                  title="Bellman optimality"
                  latex="Q^*(s,a) = r + \gamma \max_{a'} Q^*(s', a')"
                  note="Defines the best-possible action value if you act optimally from the next state onward."
                />
                <EquationCard
                  title="Tabular Q update"
                  latex="Q(s,a) \leftarrow Q(s,a) + \alpha\,(\text{target} - Q(s,a))"
                  note="Moves the table entry toward the Bellman target using learning rate alpha."
                />
                <EquationCard
                  title="Bellman target (tabular)"
                  latex="\text{target} = r + \gamma \max_{a'} Q(s', a')"
                  note="Interprets a step as two pieces: immediate reward r plus the best discounted future you could reach from s'. Gamma down-weights far-off returns so closer rewards matter more."
                />
              </div>
            </div>
          </section>

          <section className="p-5 rounded-lg bg-slate-800/70 border border-slate-700">
            <h3 className="text-2xl font-semibold mb-4">Deep Q-Network (why and what changes)</h3>
            <p className="text-slate-300 mb-4">
              Tables break when states explode (images, sensors). DQN keeps the Bellman target but replaces the table
              with a neural network that generalizes, adds a replay buffer to decorrelate data, and uses a target
              network to keep learning stable. This lets the same idea scale to visual and high-dimensional tasks.
            </p>
            <p className="text-slate-300 mb-4">
              Why it became famous: the 2015 DeepMind Atari paper showed one DQN playing dozens of Atari games from raw
              pixels, often at human level, without game-specific tweaks. The combination of replay + target network
              turned a simple Bellman update into a scalable visual learner.
            </p>
            <div className="grid md:grid-cols-2 gap-4 mb-6">
              {dqnReasons.map((item) => (
                <div key={item.title} className="p-4 rounded bg-slate-900/70 border border-slate-800">
                  <div className="font-semibold text-slate-50">{item.title}</div>
                  <p className="text-sm text-slate-200 mt-1">{item.text}</p>
                </div>
              ))}
            </div>

            <div className="grid lg:grid-cols-3 gap-4 mb-6">
              {dqnLoop.map((step) => (
                <div key={step.title} className="p-3 rounded bg-slate-900/70 border border-slate-800">
                  <div className="text-sm uppercase tracking-wide text-slate-400 font-semibold">{step.title}</div>
                  <p className="text-sm text-slate-200 mt-1">{step.detail}</p>
                </div>
              ))}
            </div>

            <div className="grid md:grid-cols-2 gap-4">
              <EquationCard
                title="Bellman target (DQN)"
                latex="y = r + \gamma \max_{a'} Q_{\text{target}}(s', a')"
                note="Same immediate-plus-future idea, but Q is produced by the frozen target network so the goalpost stays still while the online net learns."
              />
              <EquationCard
                title="DQN loss"
                latex="\mathcal{L} = \big(y - Q_{\text{online}}(s,a)\big)^2"
                note="Online network predicts Q; we minimize squared error to the target y over a replay mini-batch."
              />
            </div>

            <div className="mt-6 p-4 rounded bg-slate-900/70 border border-slate-800">
              <h4 className="text-lg font-semibold text-slate-50 mb-3">Modern uses</h4>
              <div className="grid md:grid-cols-2 gap-3 text-sm text-slate-200">
                <InfoCard
                  title="Games & simulators"
                  body="Atari (landmark), classic control (CartPole), and 3D nav tasks use DQN variants to learn from pixels."
                />
                <InfoCard
                  title="Robotics via sim-to-real"
                  body="Train in simulation with DQN-style value nets, then transfer policies to physical robots with safety checks."
                />
                <InfoCard
                  title="Recommendation & ranking"
                  body="Model user interaction as states and actions; DQN flavors optimize long-term engagement or retention."
                />
                <InfoCard
                  title="Resource management"
                  body="Cluster scheduling and caching problems are framed as RL; DQN helps choose actions that balance load and latency."
                />
              </div>
            </div>
          </section>

          <section className="p-5 rounded-lg bg-slate-800/70 border border-slate-700">
            <h3 className="text-2xl font-semibold mb-4">Hyperparameters Explained</h3>
            <p className="text-slate-300 mb-4">
              These three hyperparameters control different aspects of the learning process. Tuning them properly
              is crucial for stable and efficient learning.
            </p>
            <div className="space-y-4">
              <div id="learning-rate-section" className="p-4 rounded bg-slate-900/70 border border-slate-800 transition-all duration-500">
                <h4 className="text-lg font-semibold text-slate-50 mb-2">Learning Rate (α)</h4>
                <p className="text-sm text-slate-200 mb-3">
                  The learning rate determines how much the neural network weights change with each update. Think of it
                  as the step size when climbing down a hill toward better performance.
                </p>
                <ul className="text-sm text-slate-200 space-y-2 ml-4">
                  <li><span className="text-blue-400">•</span> <strong>Too high:</strong> Training becomes unstable, weights oscillate wildly, and the agent may never converge to a good policy.</li>
                  <li><span className="text-blue-400">•</span> <strong>Too low:</strong> Learning is very slow, requiring many more episodes to reach good performance.</li>
                  <li><span className="text-blue-400">•</span> <strong>Typical range:</strong> 0.0001 to 0.001 for DQN. Start with 0.0005 and adjust based on training stability.</li>
                </ul>
              </div>

              <div id="epsilon-decay-section" className="p-4 rounded bg-slate-900/70 border border-slate-800 transition-all duration-500">
                <h4 className="text-lg font-semibold text-slate-50 mb-2">Epsilon Decay</h4>
                <p className="text-sm text-slate-200 mb-3">
                  Controls how quickly the agent transitions from exploration (trying random actions) to exploitation
                  (using learned knowledge). The epsilon value starts at 1.0 (100% random) and decays exponentially.
                </p>
                <ul className="text-sm text-slate-200 space-y-2 ml-4">
                  <li><span className="text-blue-400">•</span> <strong>Low decay (e.g., 500):</strong> Fast transition to exploitation. Good when you want quick results but risk missing better strategies.</li>
                  <li><span className="text-blue-400">•</span> <strong>High decay (e.g., 5000):</strong> Longer exploration phase. Better for complex environments where thorough exploration is needed.</li>
                  <li><span className="text-blue-400">•</span> <strong>Formula:</strong> epsilon = epsilon_min + (1.0 - epsilon_min) * exp(-step / decay). The decay parameter is the number of steps for epsilon to drop by ~63%.</li>
                </ul>
              </div>

              <div id="gamma-section" className="p-4 rounded bg-slate-900/70 border border-slate-800 transition-all duration-500">
                <h4 className="text-lg font-semibold text-slate-50 mb-2">Gamma (γ) - Discount Factor</h4>
                <p className="text-sm text-slate-200 mb-3">
                  The discount factor determines how much the agent values future rewards compared to immediate ones.
                  It appears in the Bellman equation: Q(s,a) = r + γ * max Q(s',a').
                </p>
                <ul className="text-sm text-slate-200 space-y-2 ml-4">
                  <li><span className="text-blue-400">•</span> <strong>γ = 0:</strong> Only immediate rewards matter (myopic agent).</li>
                  <li><span className="text-blue-400">•</span> <strong>γ = 1:</strong> All future rewards weighted equally (infinite horizon).</li>
                  <li><span className="text-blue-400">•</span> <strong>Typical values:</strong> 0.95-0.99 for episodic tasks. Higher values (0.99) make the agent more patient and willing to take longer paths for better rewards.</li>
                  <li><span className="text-blue-400">•</span> <strong>Example:</strong> With γ=0.99, a reward 100 steps away is worth 0.99^100 ≈ 0.366 of its face value.</li>
                </ul>
              </div>
            </div>
          </section>

          <section className="p-5 rounded-lg bg-slate-800/70 border border-slate-700">
            <h3 className="text-xl font-semibold mb-3">Summary</h3>
            <p className="text-slate-200 text-sm leading-6 mb-3">
              Deep Q-learning is the same Bellman idea you meet in tabular Q-learning—immediate reward plus discounted
              best future reward—but implemented with a neural network so it can generalize across many states. Replay
              buffers recycle experience and break correlations; target networks keep the learning signal steady; and
              epsilon schedules balance curiosity with exploitation. Together, these upgrades let the agent learn from
              pixels or rich sensor inputs instead of tiny tables.
            </p>
            <p className="text-slate-200 text-sm leading-6 mb-3">
              Modern RL builds on this foundation. Actor-critic methods (e.g., A2C, PPO) learn both a policy and a
              value function to stabilize gradients. Policy gradient methods directly optimize the policy without
              needing an argmax over actions, which helps in continuous control. Distributional Q-learning, double/dueling
              DQN, and prioritized replay refine stability and data efficiency. The core thread remains: estimate future
              returns, improve decisions, and keep the feedback signal stable while exploring enough to discover better
              behavior.
            </p>
          </section>
        </div>
      </div>
    </section>
  );
}

function EquationCard({ title, latex, note }) {
  const html = renderToString(latex, { displayMode: true, throwOnError: false });
  return (
    <div className="p-4 rounded bg-slate-900/80 border border-slate-800 shadow-sm">
      <div className="font-semibold text-slate-50 mb-2">{title}</div>
      <div className="px-3 py-2 rounded bg-slate-950 border border-slate-800 text-center shadow-inner overflow-x-auto">
        <span
          className="text-xl font-semibold text-sky-100 inline-block"
          dangerouslySetInnerHTML={{ __html: html }}
        />
      </div>
      <p className="text-sm text-slate-200 mt-2">{note}</p>
    </div>
  );
}

function InfoCard({ title, body }) {
  return (
    <div className="p-3 rounded bg-slate-900/70 border border-slate-800">
      <div className="font-semibold text-slate-50">{title}</div>
      <p className="mt-1 text-slate-200">{body}</p>
    </div>
  );
}

function InfoRow({ label, desc }) {
  return (
    <div className="flex items-start gap-2">
      <span className="px-2 py-1 rounded bg-slate-950 border border-slate-800 text-sky-100 font-semibold text-xs min-w-[56px] text-center">
        {label}
      </span>
      <span className="text-slate-200 leading-snug">{desc}</span>
    </div>
  );
}
