using System;
using System.Collections.Generic;

class PPO
{
    private double[] policyWeights;
    private double valueWeight;
    private const double Gamma = 0.99;
    private const double ClipEpsilon = 0.2;
    private const double LearningRate = 0.01;
    private const int Epochs = 3;

    public PPO()
    {
        policyWeights = new double[] { new Random().NextDouble(), new Random().NextDouble() };
        valueWeight = new Random().NextDouble();
    }

    private double[] Policy(int state)
    {
        double logits = policyWeights[0] * state + policyWeights[1];
        double expLogits = Math.Exp(logits);
        return new double[] { expLogits / (1 + expLogits), 1 / (1 + expLogits) };
    }

    private double Value(int state)
    {
        return valueWeight * state;
    }

    private List<double> ComputeAdvantages(List<int> rewards, List<double> values)
    {
        List<double> advantages = new List<double>();
        double advantage = 0;

        for (int t = rewards.Count - 1; t >= 0; t--)
        {
            double tdError = rewards[t] + (t + 1 < values.Count ? Gamma * values[t + 1] : 0) - values[t];
            advantage = tdError + Gamma * advantage;
            advantages.Insert(0, advantage);
        }

        return advantages;
    }

    public void Train(SimpleEnv env, int episodes)
    {
        for (int episode = 0; episode < episodes; episode++)
        {
            int state = env.Reset();
            List<int> states = new List<int>();
            List<int> actions = new List<int>();
            List<int> rewards = new List<int>();

            bool done = false;
            while (!done)
            {
                double[] probs = Policy(state);
                int action = new Random().NextDouble() < probs[0] ? 0 : 1;
                var (nextState, reward, isDone) = env.Step(action);

                states.Add(state);
                actions.Add(action);
                rewards.Add(reward);
                state = nextState;
                done = isDone;
            }

            List<double> values = new List<double>();
            foreach (int s in states)
                values.Add(Value(s));

            List<double> advantages = ComputeAdvantages(rewards, values);

            for (int epoch = 0; epoch < Epochs; epoch++)
            {
                for (int i = 0; i < states.Count; i++)
                {
                    int s = states[i];
                    int a = actions[i];
                    double adv = advantages[i];

                    double[] probs = Policy(s);
                    double oldProb = probs[a];

                    double ratio = probs[a] / oldProb;
                    double clipValue = Math.Clamp(ratio, 1 - ClipEpsilon, 1 + ClipEpsilon);
                    double lossPolicy = -Math.Min(ratio * adv, clipValue * adv);

                    policyWeights[0] -= LearningRate * lossPolicy;
                    policyWeights[1] -= LearningRate * lossPolicy;

                    double targetValue = adv + Value(s);
                    double lossValue = Math.Pow(Value(s) - targetValue, 2);
                    valueWeight -= LearningRate * lossValue;
                }
            }

            Console.WriteLine($"Episode {episode + 1}: Total Reward = {rewards.Sum()}");
        }
    }

    static void Main(string[] args)
    {
        SimpleEnv env = new SimpleEnv();
        PPO ppo = new PPO();
        ppo.Train(env, 100);
    }
}
