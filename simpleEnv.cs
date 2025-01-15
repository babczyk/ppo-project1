using System;
using System.Collections.Generic;

class SimpleEnv
{
    private int state;
    public SimpleEnv()
    {
        state = 0;
    }

    public int Reset()
    {
        state = 0;
        return state;
    }

    public (int, int, bool) Step(int action)
    {
        int reward = action == 1 ? 1 : -1;
        state++;
        bool done = state >= 10;
        return (state, reward, done);
    }
}
