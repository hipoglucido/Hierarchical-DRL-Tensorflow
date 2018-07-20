gs = [0, 1, 2]
srs = [0, 1]
types = [7, 14, 21]
letters = ['A', 'B', 'C']
cancels = []
starts = []
for g in gs:
    agent = 'dqn' if g==0 else 'hdqn'
    for sr in srs:
        for letter, rs in zip(letters, types):
            filename = "g%dr%d%s.sh" % (g, sr, letter)
            cancels.append("scancel --name %s" % filename)
            starts.append("sbatch %s" % filename)
            base = "#! /bin/bash\n#SBATCH -n 1\npython main.py --agent_type=%s --scale=7000 --sparse_rewards=%d --random_seed=%d --goal_group=%d --experiment_name=singles" % (agent, sr, rs, g)
            with open(filename, 'w') as fp:
                fp.write(base)
            
print(' && '.join(starts))
print(' && '.join(cancels))

