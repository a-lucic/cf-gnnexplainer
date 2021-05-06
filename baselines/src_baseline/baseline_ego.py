from __future__ import division
from __future__ import print_function
import sys
sys.path.append('../../')
import argparse
import pickle
import numpy as np
import time
import torch
import torch.nn.functional as F
import torch_geometric.utils as g_utils

from src.gcn import GCNSynthetic
from src.utils.utils import normalize_adj, get_neighbourhood, safe_open, get_degree_matrix, create_symm_matrix_from_vec, create_vec_from_symm_matrix
from torch_geometric.utils import dense_to_sparse, k_hop_subgraph, to_dense_adj



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='syn1')

# Based on original GCN models -- do not change
parser.add_argument('--hidden', type=int, default=20, help='Number of hidden units.')
parser.add_argument('--n_layers', type=int, default=3, help='Number of convolutional layers.')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (between 0 and 1)')

# For explainer
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--keep_ego', type=int, default=0, help='0 ==> remove ego, 1 ==> keep')
parser.add_argument('--device', default='cpu', help='CPU or GPU.')
args = parser.parse_args()

print(args)

args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.autograd.set_detect_anomaly(True)


# Import dataset from GNN explainer paper
with open("../../data/gnn_explainer/{}.pickle".format(args.dataset[:4]), "rb") as f:
	data = pickle.load(f)

adj = torch.Tensor(data["adj"]).squeeze()       # Does not include self loops
features = torch.Tensor(data["feat"]).squeeze()
labels = torch.tensor(data["labels"]).squeeze()
idx_train = torch.tensor(data["train_idx"])
idx_test = torch.tensor(data["test_idx"])
edge_index = dense_to_sparse(adj)       # Needed for pytorch-geo functions

# Change to binary task: 0 if not in house, 1 if in house
if args.dataset == "syn1_binary":
	labels[labels==2] = 1
	labels[labels==3] = 1

norm_adj = normalize_adj(adj)       # According to reparam trick from GCN paper


# Set up original model, get predictions
model = GCNSynthetic(nfeat=features.shape[1], nhid=args.hidden, nout=args.hidden,
					 nclass=len(labels.unique()), dropout=args.dropout)

model.load_state_dict(torch.load("../../models/gcn_3layer_{}.pt".format(args.dataset)))
model.eval()
output = model(features, norm_adj)
y_pred_orig = torch.argmax(output, dim=1)
print("y_true counts: {}".format(np.unique(labels.numpy(), return_counts=True)))
print("y_pred_orig counts: {}".format(np.unique(y_pred_orig.numpy(), return_counts=True)))      # Confirm model is actually doing something


# Get CF examples in test set
test_cf_examples = []
start = time.time()
for i in idx_test[:]:
	best_loss = np.inf


	# Get subgraph relevant for computation
	sub_adj, sub_feat, sub_labels, node_dict = get_neighbourhood(int(i), edge_index, args.n_layers + 1, features, labels)
	new_idx = node_dict[int(i)]
	num_nodes = sub_adj.shape[0]


	# Get ego graph = 1hop subgraph
	sub_edge_index = dense_to_sparse(sub_adj)[0]
	ego_nodes, ego_edge_index, _, _ = k_hop_subgraph(new_idx.item(), 1, sub_edge_index)
	ego_adj = to_dense_adj(ego_edge_index, max_num_nodes=num_nodes).squeeze()


	# Get cf_adj, compute prediction for cf_adj
	if args.keep_ego == 0:
		cf_adj = sub_adj - ego_adj          # remove ego graph
	else:
		cf_adj = ego_adj        # keep ego graph


	# Get prediction
	A_tilde = cf_adj + torch.eye(num_nodes)

	D_tilde = get_degree_matrix(A_tilde)
	# Raise to power -1/2, set all infs to 0s
	D_tilde_exp = D_tilde ** (-1 / 2)
	D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

	# Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
	cf_norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)

	pred_cf = torch.argmax(model(sub_feat, cf_norm_adj), dim=1)[new_idx]
	pred_orig = torch.argmax(model(sub_feat, normalize_adj(sub_adj)), dim=1)[new_idx]
	loss_graph_dist = sum(sum(abs(cf_adj - sub_adj))) / 2      # Number of edges changed (symmetrical)
	print("Node idx: {}, original pred: {}, cf pred: {}, graph loss: {}".format(i, pred_orig, pred_cf, loss_graph_dist))

	if (pred_cf != pred_orig):
		cf_example = [i.item(), new_idx.item(),
			            cf_adj.detach().numpy(), sub_adj.detach().numpy(),
			            pred_cf.item(), pred_orig.item(), sub_labels[new_idx].numpy(),
			            sub_adj.shape[0], node_dict,
			               loss_graph_dist.item()]
		test_cf_examples.append(cf_example)
print("Total time elapsed: {:.4f}min".format((time.time() - start)/60))

# Save CF examples in test set

if args.keep_ego == 0:
	with safe_open("../results/remove_ego/{}_baseline_cf_examples".format(args.dataset), "wb") as f:
			pickle.dump(test_cf_examples, f)
else:
	with safe_open("../results/keep_ego/{}_baseline_cf_examples".format(args.dataset), "wb") as f:
			pickle.dump(test_cf_examples, f)
