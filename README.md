# snapshot_ensemble

Implementation of the paper [here](https://arxiv.org/pdf/1704.00109.pdf)

## How to use
In order to use this project, simply add the package to your project and then
```python
from snapshot_ensemble import SnapshotEnsemble

# <awesome pytorch NN code here>

ensemble = SnapshotEnsemble(net, criterion, restart_lr=0.1, epochs=num_epochs, batch_size=16,
                            num_snapshots=6, train_dataset=trainset, test_dataset=testset)

def closure(datum, net, crit, cuda=True, gpu=0):  # this is exactly ensemble.default_closure
    input, target = datum
    input = Variable(input).cuda(gpu, async=True) if cuda else Variable(input)
    target = Variable(target).cuda(gpu, async=True) if cuda else Variable(target)

    output = net(input)
    loss = crit(output, target)
    loss.backward()
    return loss

def forward(input, net):
    return net(input)

ensemble.train(closure=closure, print_steps=2500)
# or
# ensemble.train()  # defaults are clousre=ensemble.default_closure, and print_steps=1000


def check(output, target):
    _, predicted = torch.max(output.data, 1)
    c = (predicted == target.data).squeeze()
    return bool(c[0])

# This step takes a long time - I suggest leaving it for a while and coming back later.
ensemble.optimize_ensemble_weights(forward=ensemble.default_forward, n_iters=5, ensemble_size=3)
ensemble.save()
# ensemble.load(6)
ensemble.validate(forward=forward, ensemble_size=1, check_correctness=check)

```
It's that simple! All you need to do is define a closure (or use the default provided), a forward call (or use the default 
provided), and a way to compute test accuracy.
