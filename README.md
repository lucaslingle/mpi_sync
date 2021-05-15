# mpi_sync

Synchronized initialization for MPI runs of Pytorch projects. Works while maintaining different seeds per process, 
thus allowing reinforcement learning agents to collect distinct experience while sharing the same parameters.

```
(openai_gym) lucaslingle@Lucass-MacBook-Pro mpi_sync % mpirun -np 2 python -m unsynced
process 0, data: tensor([[-0.0024,  0.1696, -0.2603, -0.2327, -0.1218,  0.0848, -0.0063,  0.2507,
         -0.0281,  0.0837]])
process 1, data: tensor([[ 0.1629, -0.1396, -0.0613,  0.1484, -0.2977,  0.1896, -0.0651,  0.1609,
          0.0440, -0.0387]])
process 1, data: tensor([0.0877])
process 0, data: tensor([-0.0956])
(openai_gym) lucaslingle@Lucass-MacBook-Pro mpi_sync % 
(openai_gym) lucaslingle@Lucass-MacBook-Pro mpi_sync % mpirun -np 2 python -m synced  
process 0, data: tensor([[-0.0024,  0.1696, -0.2603, -0.2327, -0.1218,  0.0848, -0.0063,  0.2507,
         -0.0281,  0.0837]])
process 1, data: tensor([[-0.0024,  0.1696, -0.2603, -0.2327, -0.1218,  0.0848, -0.0063,  0.2507,
         -0.0281,  0.0837]])
process 0, data: tensor([-0.0956])
process 1, data: tensor([-0.0956])
```

Proof of concept only. Complete implementation of a parallelized RL agent coming soon.
