# Description

This is the official repository of the **'Batch Bayesian Optimization with adaptive batch acquisition functions via multi-objective optimization'** by Jixiang Chen, Fu Luo, Genghui Li, and Zhenkun Wang.

In this study, we propose a novel Batch Bayesian Optimization method with adaptive batch acquisition functions via multi-objective optimization (BBO-DMoAF). Specifically, multiple acquisition functions are adaptively chosen to form a multi-objective optimization problem (MOP). Its Pareto-optimal solutions provide the candidate solutions for expensive evaluation according to a minimum-diverse-exploitative (MDE) strategy. The experimental results show the advantages of the proposed BBO-ABAFMo over some state-of-the-art methods.

# Getting started

This code is implemented under Window 10 and Python 3.8 environment.

To set up the required environment, firstly install packages in 'requirements.txt'.

Then, find the package of 'platypus' in your environment (the path should be Your_env_path/Lib/site-packages/platypus), and replace the file 'algorithms.py' and 'core.py' with our provided ones. This is because we have done some modification to the package's source code to fix the replication solutions bug.

# License

The source code of this repository is released only for academic use. See the [license](LICENSE) file for details.

# Acknowledgement

This work is supported by the National Natural Science Foundation of China (Grant No. 62106096, 62206120), Characteristic Innovation Project of Colleges and Universities in Guangdong Province (Grant No. 2022KTSCX110), Shenzhen Technology Plan (Grant No. JCYJ202205301130130311), and Special Funds for the Cultivation of Guangdong College Students’ Scientific and Technological Innovation (‘‘Climbing Program’’ Special Funds)(Grant No. pdjh2022c0093).

# Citation

If you find this work or code helpful in your project, please cite

```

```

