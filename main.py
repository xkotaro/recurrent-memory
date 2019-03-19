from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable

from datagenerator import DelayedEstimationTask
from model import RecurrentNet


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target, si, opt) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        _, output = model(data)
        loss = torch.nn.MSELoss(output[-25:], target[-25:])
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target, si, opt in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += torch.nn.MSELoss(output[-25:], target[-25:])
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = DelayedEstimationTask(max_iter=25000, n_loc=1, n_in=50, n_out=50, stim_dur=25, delay_dur=100,
                                          resp_dur=25, kappa=2.0, spon_rate=0.001, transform=transform)
    test_dataset = DelayedEstimationTask(max_iter=2500, n_loc=1, n_in=50, n_out=50, stim_dur=25, delay_dur=100,
                                         resp_dur=25, kappa=2.0, spon_rate=0.001, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, args.batch_size)

    model = RecurrentNet(n_in=50, n_hid=500, n_out=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)

    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), "recurrent_memory.pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN training')
    parser.add_argument('--batch_size', type=int, default=50, metavar='N',
                        help='input batch size for training (default: 50)')
    parser.add_argument('--test_batch_size', type=int, default=50, metavar='N',
                        help='input batch size for testing (default: 50)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    print(args)
    main()