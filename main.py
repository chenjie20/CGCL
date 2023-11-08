import time
import os.path as osp
import argparse
import warnings

import torch
import torch_geometric.transforms as T
from torch_geometric.utils import add_self_loops, negative_sampling


from utils import *
from models import *
from layers import *
from loss import *
from dataprocessing import *

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='CGCL')

parser.add_argument('--dataset', type=str, default='Cora',
                    choices=['Cora', 'Citeseer', 'Pubmed', 'Photo', 'Computers'],
                    help='Dataset name')
parser.add_argument('--ratio', type=float, default=0.1, help='The ratio of testing edges dropped.')
parser.add_argument('--seed', type=int, default=10, help='Initializing random seed.')
parser.add_argument('--dim_hidden_feature', type=int, default=512, help='dimensionality of hidden representation.')
parser.add_argument("--epochs", default=800, help='Number of epochs to training.')
parser.add_argument('--runs', type=int, default=2, help='Number of runs.')
parser.add_argument('--learning_rate', '-lr', type=float, default=1e-2,
                    help='Initializing learning rate chosen from [1e-3, 5e-3, 1e-2, 5e-2]')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Initializing weight decay.')
parser.add_argument('--gpu', default=0, type=int, help='GPU device idx.')

args = parser.parse_args()
print("==========\nArgs:{}\n==========".format(args))

torch.cuda.set_device(args.gpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    set_seed(args.seed)

    def train(epoch):
        model.train()
        x, edge_index = train_data.x, train_data.edge_index
        edge_views = generate_two_views(edge_index)
        assert len(edge_views) == 2
        aug_edge_index, _ = add_self_loops(edge_index)

        for idx in range(len(edge_views)):
            v_idx = 1 if idx == 0 else 0
            neg_edges = negative_sampling(
                aug_edge_index,
                num_nodes=train_data.num_nodes,
                num_neg_samples=edge_views[v_idx].view(2, -1).size(1),
            ).view_as(edge_views[v_idx])

            z = model(x, edge_views[idx])
            pos_out = model.edge_decoder(
                z, edge_views[v_idx], sigmoid=False
            )
            neg_out = model.edge_decoder(z, neg_edges, sigmoid=False)
            loss = binary_cross_entropy_loss(pos_out, neg_out)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

        if epoch % 50 == 0 or epoch == args.epochs - 1:
            print('training, epoch {}, Loss:{:.7f}'.format(epoch, loss.item()))

        return loss.item()


    @torch.no_grad()
    def test():
        model.eval()
        z = model(train_data.x, train_data.edge_index)

        pos_pred = model.edge_decoder(z, val_data.pos_edge_label_index).squeeze().cpu()
        neg_pred = model.edge_decoder(z, val_data.neg_edge_label_index).squeeze().cpu()
        valid_auc, valid_ap = testing(pos_pred, neg_pred)

        pos_pred = model.edge_decoder(z, test_data.pos_edge_label_index).squeeze().cpu()
        neg_pred = model.edge_decoder(z, test_data.neg_edge_label_index).squeeze().cpu()
        test_auc, test_ap = testing(pos_pred, neg_pred)

        all_results = {'AUC': (valid_auc, test_auc), 'AP': (valid_ap, test_ap)}

        return all_results


    transform = T.Compose([
        T.ToUndirected(),
        T.ToDevice(device),
    ])
    loggers = {
        'AUC': Logger(args.runs, args),
        'AP': Logger(args.runs, args),
    }

    dataset = load_data(args.dataset)
    data = transform(dataset[0])

    ratios = [0.1, 0.2]
    # learning_rates = np.array([1e-3, 5e-3, 0.01, 0.05], dtype=np.float32)
    # dims_layers = np.array([512, 256, 128, 64], dtype=np.int32)

    for ratio_idx in range(len(ratios)):
        args.ratio = ratios[ratio_idx]
        train_data, val_data, test_data = T.RandomLinkSplit(num_val=args.ratio/2, num_test=args.ratio,
                                                            is_undirected=True,
                                                            split_labels=True,
                                                            add_negative_train_samples=True)(data)
        train_data = train_data.to(device)
        val_data = val_data.to(device)
        test_data = test_data.to(device)

        # for lr_idx in range(learning_rates.shape[0]):
        #     args.learning_rate = learning_rates[lr_idx]
        #     for dim_idx in range(dims_layers.shape[0]):
        #         args.dim_hidden_feature = dims_layers[dim_idx]
        for run in range(args.runs):
            edge_decoder = EdgeDecoder(args.dim_hidden_feature)
            model = CGCL(edge_decoder, data.num_features, args.dim_hidden_feature)
            model = model.to(device)
            model.reset_parameters()

            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=args.learning_rate,
                                         weight_decay=args.weight_decay)

            for epoch in range(args.epochs):
                t1 = time.time()
                loss = train(epoch)
                t2 = time.time()

            results = test()

            for key, result in results.items():
                valid_result, test_result = result
                print(key)
                print(f'--Testing on Run: {run + 1:02d}, '
                      f'Valid: {valid_result:.2%}, '
                      f'Test: {test_result:.2%}')

            for key, result in results.items():
                loggers[key].add_result(run, result)

        print('--Final result')
        for key in loggers.keys():
            print(key)
            mean_result, std_result = loggers[key].print_statistics()

            with open('final_mean_results_%s_%s.txt' % (args.dataset, args.ratio), 'a+') as f:
                f.write('{:.2f} \t {:.4f}  \t {} \t  {} \t {:.4f} \t {:.2f}'
                        '\n'.format(args.ratio, args.learning_rate, args.dim_feature, key, mean_result, std_result))
                f.flush()
