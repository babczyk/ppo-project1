using System;
using System.Collections.Generic;

// See https://aka.ms/new-console-template for more information
GridWorldEnv env = new GridWorldEnv();
PPO ppo = new PPO();
ppo.Train(env, 5000);
Console.WriteLine("Training complete");