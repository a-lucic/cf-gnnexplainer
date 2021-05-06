# tried to retrain our GCN using edge index instead of adj


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
import torch.optim as optim
from torch_geometric.utils import accuracy
from torch.nn.utils import clip_grad_norm
# from gnn_explainer.explainer import explain
# from gnnexplainer import GNNExplainer
from torch_geometric.nn import GNNExplainer


from src.gcn import GCNSynthetic, GCNSynthetic_v2
from src.utils.utils import normalize_adj, get_neighbourhood, safe_open, get_degree_matrix, create_symm_matrix_from_vec, create_vec_from_symm_matrix
from torch_geometric.utils import dense_to_sparse



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='syn1')

# Based on original GCN models -- do not change
parser.add_argument('--hidden', type=int, default=20, help='Number of hidden units.')
parser.add_argument('--n_layers', type=int, default=3, help='Number of convolutional layers.')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (between 0 and 1)')

# For explainer
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--num_epochs', type=int, default=500, help='Num epochs for explainer')
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




# Recreate original model using edge_index instead of adj so we can use PyTorch Geo implementation of GNNExplainer
# Use exact same hyperparamter settings we used to train the original GCNs
model_v2 = GCNSynthetic_v2(nfeat=features.shape[1], nhid=args.hidden, nout=args.hidden,
					 nclass=len(labels.unique()), dropout=args.dropout)
if args.dataset == "syn1":      # ba-shapes
	optimizer = optim.Adam(model_v2.parameters(), lr=0.01, weight_decay=0.001)
elif args.dataset == "syn4":       # tree-cycles
	optimizer = optim.Adam(model_v2.parameters(), lr=0.001, weight_decay=0.001)
if args.dataset == "syn5":          # tree-grid
	optimizer = optim.Adam(model_v2.parameters(), lr=0.001, weight_decay=0.001)


def train(epoch):
	t = time.time()
	model_v2.train()
	optimizer.zero_grad()
	output = model_v2(features, edge_index[0])
	loss_train = model_v2.loss(output[idx_train], labels[idx_train])
	y_pred = torch.argmax(output, dim=1)
	acc_train = accuracy(y_pred[idx_train], labels[idx_train])
	loss_train.backward()
	clip_grad_norm(model_v2.parameters(), 2.0)      # same as in original GCN models
	optimizer.step()
	print('Epoch: {:04d}'.format(epoch+1),
		  'loss_train: {:.4f}'.format(loss_train.item()),
		  'acc_train: {:.4f}'.format(acc_train),
		  'time: {:.4f}s'.format(time.time() - t))

def test():
	model_v2.eval()
	output = model_v2(features, edge_index[0])
	loss_test = F.nll_loss(output[idx_test], labels[idx_test])
	y_pred = torch.argmax(output, dim=1)
	acc_test = accuracy(y_pred[idx_test], labels[idx_test])
	print("Test set results:",
		  "loss= {:.4f}".format(loss_test.item()),
		  "accuracy= {:.4f}".format(acc_test))
	return y_pred

for epoch in range(1000):       # original GCNs were trained for 1000 epochs (like in GNNExplainer paper)
	train(epoch)

# Testing
y_pred = test()





output_v2 = model_v2(features, edge_index[0])
y_pred_orig_v2 = torch.argmax(output_v2, dim=1)
print("y_pred_orig_v2 counts: {}".format(np.unique(y_pred_orig_v2.numpy(), return_counts=True)))      # Confirm model is actually doing something





# Get CF examples in test set
# test_cf_examples = []
# start = time.time()
# for i in idx_test[:]:
#
# 	sub_adj, sub_feat, sub_labels, node_dict = get_neighbourhood(int(i), edge_index, args.n_layers + 1, features,
# 																 labels)
# 	new_idx = node_dict[int(i)]
#
#
# 	explainer = GNNExplainer(model, epochs=args.num_epochs)
# 	# _, edge_mask = explainer.explain_node(10, x=sub_feat, adj=sub_adj, edge_index=edge_index[0])
# 	_, edge_mask = explainer.explain_node(10, x=features, adj=norm_adj, edge_index=edge_index[0])

	# Create explainer
	# explainer = explain.Explainer(
	# 	model=model,
	# 	adj=adj,
	# 	feat=features,
	# 	label=labels,
	# 	pred=y_pred_orig,
	# 	train_idx=idx_train,
		# args=prog_args,
		# writer=writer,
		# print_training=True,
		# graph_mode=graph_mode,
		# graph_idx=prog_args.graph_idx,
	# )


#
#
# 	best_loss = np.inf
#
# 	for n in range(args.num_epochs):
# 		sub_adj, sub_feat, sub_labels, node_dict = get_neighbourhood(int(i), edge_index, args.n_layers + 1, features, labels)
# 		new_idx = node_dict[int(i)]
#
# 		# Get CF adj, new prediction
# 		num_nodes = sub_adj.shape[0]
#
# 		# P_hat needs to be symmetric ==> learn vector representing entries in upper/lower triangular matrix and use to populate P_hat later
# 		P_vec_size = int((num_nodes * num_nodes - num_nodes) / 2)  + num_nodes
#
# 		# Randomly initialize P_vec in [-1, 1]
# 		r1 = -1
# 		r2 = 1
# 		P_vec = torch.FloatTensor((r1 - r2) * torch.rand(P_vec_size) + r2)
# 		P_hat_symm = create_symm_matrix_from_vec(P_vec, num_nodes)      # Ensure symmetry
# 		P = (F.sigmoid(P_hat_symm) >= 0.5).float()      # threshold P_hat
#
# 		# Get cf_adj, compute prediction for cf_adj
# 		cf_adj = P * sub_adj
# 		A_tilde = cf_adj + torch.eye(num_nodes)
#
# 		D_tilde = get_degree_matrix(A_tilde)
# 		# Raise to power -1/2, set all infs to 0s
# 		D_tilde_exp = D_tilde ** (-1 / 2)
# 		D_tilde_exp[torch.isinf(D_tilde_exp)] = 0
#
# 		# Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
# 		cf_norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)
#
# 		pred_cf = torch.argmax(model(sub_feat, cf_norm_adj), dim=1)[new_idx]
# 		pred_orig = torch.argmax(model(sub_feat, normalize_adj(sub_adj)), dim=1)[new_idx]
# 		loss_graph_dist = sum(sum(abs(cf_adj - sub_adj))) / 2      # Number of edges changed (symmetrical)
# 		print("Node idx: {}, original pred: {}, cf pred: {}, graph loss: {}".format(i, pred_orig, pred_cf, loss_graph_dist))
#
# 		if (pred_cf != pred_orig) & (loss_graph_dist < best_loss):
# 			best_loss = loss_graph_dist
# 			print("best loss: {}".format(best_loss))
# 			best_cf_example = [i.item(), new_idx.item(),
# 							cf_adj.detach().numpy(), sub_adj.detach().numpy(),
# 							pred_cf.item(), pred_orig.item(), sub_labels[new_idx].numpy(),
# 							sub_adj.shape[0], node_dict,
# 							   loss_graph_dist.item()]
# 	test_cf_examples.append(best_cf_example)
# 	print("Time for {} epochs of one example: {:.4f}min".format(args.num_epochs, (time.time() - start)/60))
# print("Total time elapsed: {:.4f}min".format((time.time() - start)/60))
#
# # Save CF examples in test set
# with safe_open("../results/random_perturb/{}_baseline_cf_examples_epochs{}".format(args.dataset, args.num_epochs), "wb") as f:
# 		pickle.dump(test_cf_examples, f)
