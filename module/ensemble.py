#
# Created by Aman LaChapelle on 7/5/17.
#
# snapshot-ensemble
# Copyright (c) 2017 Aman LaChapelle
# Full license at snapshot-ensemble/LICENSE.txt
#

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import multiprocessing as mp
import numpy as np
from typing import List, Callable


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
                 has_hidden_states: bool=False  # If the network has hidden states. See the note about net if it does
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
        self.has_hidden_states = has_hidden_states

        if self.has_hidden_states:
            assert hasattr(self.net, "init_hidden")
            assert hasattr(self.net, "reset_hidden")

        if self.use_cuda:
            self.net = self.net.cuda()

        self.snapshots = []

    def _a_t(self, step_no: int) -> float:
        return self.a0/2. * (np.cos((np.pi*(step_no % self.ToM))/self.ToM) + 1)

    @staticmethod
    def _ensemble_vote(votes: List[Variable]) -> Variable:

        outcome = torch.stack(votes, dim=0)  # stack it up
        outcome = torch.mean(outcome, dim=0)  # compute mean along the right axis
#         outcome = np.mean(outcome.cpu().data.numpy(), axis=0)  # compute mean along the right axis

#         outcome = Variable(torch.from_numpy(outcome).float())  # bring us back to pytorch

        return outcome

    def train(self, print_steps: int=1000) -> None:

        step_counter = 0
        snapshot_counter = 1
        running_loss = 0.0
        for epoch in range(self.epochs):
            for j, data in enumerate(self.data_loader):
                input, target = data
                input = Variable(input).cuda() if self.use_cuda else Variable(input)
                target = Variable(target).cuda() if self.use_cuda else Variable(target)

                opt = optim.SGD(self.net.parameters(), lr=self._a_t(step_counter))  # function starts at zero
                opt.zero_grad()

                output = None
                model_hidden = None
                if self.has_hidden_states:
                    model_hidden = self.net.init_hidden()
                    model_hidden = model_hidden.cuda() if self.use_cuda else model_hidden
                    output, model_hidden = self.net(input, model_hidden)
                else:
                    output = self.net(input)

                loss = self.criterion(output, target)
                loss.backward()
                running_loss += loss.data[0]
                opt.step()

                if self.has_hidden_states:
                    self.net.reset_hidden(model_hidden)  # reset the model's hidden data

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

    def validate(self, ensemble_size: int=None, check_correctness: Callable=None) -> float:

        if not ensemble_size:
            ensemble_size = self.M  # could be less than M, in which case we use the last ones
        else:
            assert ensemble_size <= self.M  # Must be less than M

        total_correct = 0
        for j, data in enumerate(self.test_loader):
            input, target = data
            input = Variable(input).cuda() if self.use_cuda else Variable(input)
            target = Variable(target).cuda() if self.use_cuda else Variable(target)

            if ensemble_size > 1:
                output = []  # size is (ensemble x output_tensor.size()) <- batch_size = 1
                for m in range(1, ensemble_size):
                    self.net.load_state_dict(self.snapshots[-m])
                    if self.has_hidden_states:
                        model_hidden = self.net.init_hidden()
                        out, model_hidden = self.net(input, model_hidden)
                        self.net.reset_hidden(model_hidden)  # reset the model's hidden data
                    else:
                        out = self.net(input)

                    output.append(out)

                output = self._ensemble_vote(output)
            else:
                self.net.load_state_dict(self.snapshots[-1])
                if self.has_hidden_states:
                    model_hidden = self.net.init_hidden()
                    output, model_hidden = self.net(input, model_hidden)
                    self.net.reset_hidden(model_hidden)  # reset the model's hidden data
                else:
                    output = self.net(input)

            total_correct += 1 if check_correctness(output.cuda(), target.cuda()) else 0

        print("Ensemble Size: %d - Percent Correct: %.2f" %
              (ensemble_size, float(total_correct/len(self.test_loader)) * 100))

        return float(total_correct/len(self.test_loader))

    def test(self, data_loader: DataLoader, ensemble_size: int=None, check_correctness: Callable=None):
        if not ensemble_size:
            ensemble_size = self.M  # could be less than M, in which case we use the last ones
        else:
            assert ensemble_size <= self.M  # Must be less than M

        total_correct = 0
        for j, data in enumerate(data_loader):
            input, target = data
            input = Variable(input).cuda() if self.use_cuda else Variable(input)
            target = Variable(target).cuda() if self.use_cuda else Variable(target)

            output = []  # size is (ensemble x output_tensor.size()) <- batch_size = 1
            for m in range(1, ensemble_size):
                self.net.load_state_dict(self.snapshots[-m])
                if self.has_hidden_states:
                    model_hidden = self.net.init_hidden()
                    out, model_hidden = self.net(input, model_hidden)
                    self.net.reset_hidden(model_hidden)  # reset the model's hidden data
                else:
                    out = self.net(input)
                output.append(out)

            if ensemble_size > 1:
                output = self._ensemble_vote(output)
            else:
                output = output[0]

            total_correct += 1 if check_correctness(output.cuda(), target.cuda()) else 0

        print("Percent Correct: {:.2f}".format(float(total_correct/len(self.test_loader)) * 100))

        return float(total_correct/len(self.test_loader))

    def save(self) -> None:
        for i, snapshot in enumerate(self.snapshots):
            torch.save(snapshot, "snapshot_%d.dat" % (i+1))

    def load(self, num_snapshots: int) -> None:
        for i in range(1, num_snapshots+1):
            snapshot = torch.load("snapshot_%d.dat" % i)
            self.snapshots.append(snapshot)

