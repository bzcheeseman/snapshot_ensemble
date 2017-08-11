#
# Created by Aman LaChapelle on 7/5/17.
#
# snapshot_ensemble
# Copyright (c) 2017 Aman LaChapelle
# Full license at snapshot_ensemble/LICENSE.txt
#

import torch
import torch.nn as nn
import torch.nn.functional as Funct
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import multiprocessing as mp
import numpy as np
from typing import List, Callable, Tuple


# See https://arxiv.org/pdf/1704.00109.pdf for reference
class SnapshotEnsemble(object):
    def __init__(self,
                 net: nn.Module,  # If it has hidden states, it must have init_hidden and reset_hidden methods
                 criterion: nn.Module,  # An instance of a criterion that's appropriate for the data being used
                 restart_lr: float,  # The starting lr (\alpha_0 in the paper)
                 epochs: int,  # The total number of epochs
                 batch_size: int,  # Batch size for the data loader
                 num_snapshots: int,  # The number of snapshots you want in an ensemble
                 train_dataset: Dataset,  # The dataset to use for training
                 test_dataset: Dataset=None,  # The dataset to use for validation/testing
                 use_cuda: bool=torch.cuda.is_available(),  # Whether or not to use CUDA
                 gpu: int=0
                 ) -> None:

        self.net = net
        self.criterion = criterion
        self.a0 = restart_lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.data_loader = DataLoader(train_dataset,
                                      batch_size=self.batch_size, shuffle=True, num_workers=min([mp.cpu_count(), 4]))
        self.test_loader = DataLoader(test_dataset, batch_size=1,
                                      shuffle=False, num_workers=2) if test_dataset else None

        self.T = len(self.data_loader) * epochs
        self.M = num_snapshots
        self.ToM = np.floor(self.T/self.M)

        self.use_cuda = use_cuda
        self.gpu = gpu
        self.ensemble_weights = torch.ones(self.M)

        if self.use_cuda:
            self.net = self.net.cuda()

        self.snapshots = []

    def _a_t(self, step_no: int) -> float:
        return self.a0/2. * (np.cos((np.pi*(step_no % self.ToM))/self.ToM) + 1)

    @staticmethod
    def default_closure(datum: Tuple,
                        net: nn.Module,
                        crit: nn.Module,
                        cuda: bool=True,
                        gpu: int=0) -> Variable:
        input, target = datum
        input = Variable(input).cuda(gpu, async=True) if cuda else Variable(input)
        target = Variable(target).cuda(gpu, async=True) if cuda else Variable(target)

        output = net(input)
        loss = crit(output, target)
        loss.backward()
        return loss

    @staticmethod
    def default_forward(input: Variable,
                        net: nn.Module) -> Variable:
        return net(input)

    def optimize_ensemble_weights(self,
                                  forward: Callable=default_forward,
                                  n_iters: int=1,
                                  ensemble_size: int=6) -> None:

        if ensemble_size < self.M:
            print("Running optimize with ensemble size = {}".format(ensemble_size))
            print("This limits the ensemble size you can use in validate or test with the optimized weights "
                  "to be <= {}".format(ensemble_size))

        min_loss = 1e6
        for iter in range(n_iters):
            weights = nn.Parameter(torch.rand(ensemble_size).cuda(self.gpu), requires_grad=True) if self.use_cuda \
                else nn.Parameter(torch.rand(ensemble_size), requires_grad=True)
            opt = optim.LBFGS([weights], lr=0.001)
            crit = nn.CrossEntropyLoss()

            total_loss = 0
            for j, data in enumerate(self.test_loader):
                input, target = data
                input = Variable(input).cuda(self.gpu, async=True) if self.use_cuda else Variable(input)
                target = Variable(target).cuda(self.gpu, async=True) if self.use_cuda else Variable(target)

                def cl():
                    opt.zero_grad()
                    output = []  # size is (ensemble x output_tensor.size()) <- batch_size = 1
                    for m in range(1, ensemble_size+1):  # guaranteed to run at least once
                        self.net.load_state_dict(self.snapshots[-m])
                        out = forward(input, self.net)
                        output.append(out)
                    Funct.softmax(weights)
                    outcome = self._ensemble_vote(weights, output)
                    loss = crit(outcome, target)
                    loss.backward()
                    return loss

                opt.step(cl)
                total_loss += cl().data[0]

            if total_loss/len(self.test_loader) < min_loss:
                min_loss = total_loss
                self.ensemble_weights = Variable(weights.data)

        print("Optimal weights: ", self.ensemble_weights.data)

    @staticmethod
    def _ensemble_vote(weights: Variable,
                       votes: List[Variable]) -> Variable:
        outcome = torch.cat(votes, 0)  # stack it up
        # compute mean along the right axis
        outcome = torch.mean(weights.unsqueeze(1).repeat(1, outcome.size(1)) * outcome, dim=0)

        return outcome

    def train(self,
              closure: Callable=default_closure,
              # Signature matches:
              # closure(Tuple[FloatTensor...], Module, Module, bool, int) -> Variable
              print_steps: int=1000) -> None:

        step_counter = 0
        snapshot_counter = 1
        running_loss = 0.0
        for epoch in range(self.epochs):
            for j, data in enumerate(self.data_loader):

                opt = optim.SGD(self.net.parameters(), lr=self._a_t(step_counter))  # function starts at zero
                opt.zero_grad()

                loss = closure(data, self.net, self.criterion, self.use_cuda, self.gpu)
                running_loss += loss.data[0]
                opt.step()

                if step_counter == self.ToM * snapshot_counter:
                    print("================== Saved Snapshot %d ==================" % snapshot_counter)
                    torch.save(self.net.state_dict(), "snapshot_%d.dat" % snapshot_counter)
                    self.snapshots.append(torch.load("snapshot_%d.dat" % snapshot_counter))
                    snapshot_counter += 1

                if (j % print_steps) == 0:
                    print("Current Step: %d - Average Loss: %.4f - Training Snapshot: %d" %
                          (step_counter, running_loss/print_steps, snapshot_counter))

                    running_loss = 0.0

                step_counter += 1

    def validate(self,
                 forward: Callable=default_forward,
                 ensemble_size: int=None,
                 use_weights: bool=True,
                 check_correctness: Callable=None) -> float:

        if not ensemble_size:
            ensemble_size = self.M  # could be less than M, in which case we use the last ones
        else:
            assert ensemble_size <= self.M  # Must be less than M

        total_correct = 0
        for j, data in enumerate(self.test_loader):
            input, target = data
            input = Variable(input).cuda(self.gpu, async=True) if self.use_cuda else Variable(input)
            target = Variable(target).cuda(self.gpu, async=True) if self.use_cuda else Variable(target)

            output = []  # size is (ensemble x output_tensor.size()) <- batch_size = 1
            for m in range(1, ensemble_size+1):  # guaranteed to run at least once
                self.net.load_state_dict(self.snapshots[-m])
                out = forward(input, self.net)
                output.append(out)

            if ensemble_size > 1 and use_weights:
                output = self._ensemble_vote(self.ensemble_weights, output)
            elif ensemble_size > 1 and not use_weights:
                ones = Variable(torch.ones(ensemble_size)).cuda(self.gpu) if self.use_cuda \
                    else Variable(torch.ones(ensemble_size))
                output = self._ensemble_vote(ones, output)
            else:
                output = output[0]

            total_correct += 1 if check_correctness(output, target) else 0

        print("Ensemble Size: %d - Percent Correct: %.2f" %
              (ensemble_size, float(total_correct/len(self.test_loader)) * 100))

        return float(total_correct/len(self.test_loader))

    def test(self,
             data_loader: DataLoader,
             forward: Callable = default_forward,
             ensemble_size: int=None,
             use_weights: bool=True,
             check_correctness: Callable=None) -> float:
        if not ensemble_size:
            ensemble_size = self.M  # could be less than M, in which case we use the last ones
        else:
            assert ensemble_size <= self.M  # Must be less than M

        total_correct = 0
        for j, data in enumerate(data_loader):
            input, target = data
            input = Variable(input).cuda(self.gpu, async=True) if self.use_cuda else Variable(input)
            target = Variable(target).cuda(self.gpu, async=True) if self.use_cuda else Variable(target)

            output = []  # size is (ensemble x output_tensor.size()) <- batch_size = 1
            for m in range(1, ensemble_size+1):  # guaranteed to run at least once
                self.net.load_state_dict(self.snapshots[-m])
                out = forward(input, self.net)
                output.append(out)

            if ensemble_size > 1 and use_weights:
                output = self._ensemble_vote(self.ensemble_weights, output)
            elif ensemble_size > 1 and not use_weights:
                ones = Variable(torch.ones(ensemble_size)).cuda(self.gpu) if self.use_cuda \
                    else Variable(torch.ones(ensemble_size))
                output = self._ensemble_vote(ones, output)
            else:
                output = output[0]

            total_correct += 1 if check_correctness(output, target) else 0

        print("Percent Correct: {:.2f}".format(float(total_correct/len(self.test_loader)) * 100))

        return float(total_correct/len(self.test_loader))

    def save(self) -> None:
        for i, snapshot in enumerate(self.snapshots):
            torch.save(snapshot, "snapshot_%d.dat" % (i+1))
        torch.save(self.ensemble_weights, "optimal_weights.dat")

    def load(self, num_snapshots: int) -> None:
        for i in range(1, num_snapshots+1):
            snapshot = torch.load("snapshot_%d.dat" % i)
            self.snapshots.append(snapshot)
        self.ensemble_weights = torch.load("optimal_weights.dat")

