# snapshot_ensemble

Implementation of the paper [here](https://arxiv.org/pdf/1704.00109.pdf)

## How to use
In order to use this project, simply add the package to your project and then
```python
from snapshot_ensemble import SnapshotEnsemble

<awesome pytorch NN code here>
```
It's that simple! All you need to do is define a closure (or use the default provided) and a way to compute test
accuracy. If you look at test.py, the last 11 lines are all you need to add to use a snapshot ensemble in your
project.