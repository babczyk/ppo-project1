using System;
using System.Collections.Generic;
class GridWorldEnv
{
    private int[,] grid;
    private (int row, int col) agentPos;
    private (int row, int col) goalPos;
    private Random random;
    private const int EMPTY = 0;
    private const int WALL = 1;
    private const int AGENT = 2;
    private const int GOAL = 3;
    private const int SIZE = 10;

    public GridWorldEnv()
    {
        random = new Random();
        grid = new int[SIZE, SIZE];
        InitializeGrid();
    }

    private void InitializeGrid()
    {
        // Clear grid
        for (int i = 0; i < SIZE; i++)
            for (int j = 0; j < SIZE; j++)
                grid[i, j] = EMPTY;

        // Place walls randomly
        int numWalls = SIZE * 2;
        for (int i = 0; i < numWalls; i++)
        {
            int row = random.Next(SIZE);
            int col = random.Next(SIZE);
            if (grid[row, col] == EMPTY)
                grid[row, col] = WALL;
        }

        // Place agent in random empty position
        do
        {
            agentPos.row = random.Next(SIZE);
            agentPos.col = random.Next(SIZE);
        } while (grid[agentPos.row, agentPos.col] != EMPTY);
        grid[agentPos.row, agentPos.col] = AGENT;

        // Place goal in random empty position
        do
        {
            goalPos.row = random.Next(SIZE);
            goalPos.col = random.Next(SIZE);
        } while (grid[goalPos.row, goalPos.col] != EMPTY);
        grid[goalPos.row, goalPos.col] = GOAL;
    }

    public int GetState()
    {
        // Encode state as: relative distance to goal
        int rowDiff = goalPos.row - agentPos.row + SIZE;
        int colDiff = goalPos.col - agentPos.col + SIZE;
        return rowDiff * (2 * SIZE + 1) + colDiff;
    }

    public int Reset()
    {
        InitializeGrid();
        return GetState();
    }

    public (int, int, bool) Step(int action, int totalReward, int episode, int state, List<int> rewards)
    {
        // Actions: 0=up, 1=right, 2=down, 3=left
        int reward = -1; // Small negative reward for each step
        bool done = false;

        // Calculate new position
        (int row, int col) newPos = agentPos;
        switch (action)
        {
            case 0: newPos.row--; break; // Up
            case 1: newPos.col++; break; // Right
            case 2: newPos.row++; break; // Down
            case 3: newPos.col--; break; // Left
        }


        int currentDistance = Math.Abs(agentPos.row - goalPos.row) + Math.Abs(agentPos.col - goalPos.col);
        int newDistance = Math.Abs(newPos.row - goalPos.row) + Math.Abs(newPos.col - goalPos.col);
        Console.WriteLine(@"currentDistance: " + currentDistance + " newDistance: " + newDistance);

        // Check if new position is valid
        if (newPos.row >= 0 && newPos.row < SIZE &&
            newPos.col >= 0 && newPos.col < SIZE &&
            grid[newPos.row, newPos.col] != WALL)
        {
            // Update grid
            grid[agentPos.row, agentPos.col] = EMPTY;
            agentPos = newPos;

            // Check if reached goal
            if (grid[agentPos.row, agentPos.col] == GOAL)
            {
                reward = 500;
                done = true;
            }
            else if (newDistance <= currentDistance)
            {
                reward = 2; // Bonus for getting closer to the goal
            }

            grid[agentPos.row, agentPos.col] = AGENT;
        }
        else
        {
            reward = -1; // Penalty for hitting wall or boundary
        }


        // End episode if taking too long
        if (GetStepCount() > SIZE * SIZE * 2)
        {

            reward = -10;
            stepCount = 0;
        }

        Render(totalReward, episode, state, rewards);
        return (GetState(), reward, done);
    }

    private int stepCount = 0;
    private int GetStepCount()
    {
        return ++stepCount;
    }

    public void Render(int totalReward, int episode, int state, List<int> rewards)
    {
        Console.Clear();
        for (int i = 0; i < SIZE; i++)
        {
            for (int j = 0; j < SIZE; j++)
            {
                switch (grid[i, j])
                {
                    case EMPTY: Console.Write(" . "); break;
                    case WALL: Console.Write(" # "); break;
                    case AGENT: Console.Write(" A "); break;
                    case GOAL: Console.Write(" G "); break;
                }
            }
            Console.WriteLine();
        }
        Console.Write(@"total reward: " + totalReward + " episode: " + episode + " state: " + state);

        Thread.Sleep(100); // Slow down visualization
    }
}