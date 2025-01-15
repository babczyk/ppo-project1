using System;
using System.Collections.Generic;

// See https://aka.ms/new-console-template for more information
SimpleEnv env = new SimpleEnv();
PPO ppo = new PPO();
ppo.Train(env, 100);
Console.WriteLine("Training complete");