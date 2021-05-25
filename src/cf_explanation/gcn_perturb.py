import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from utils.utils import get_degree_matrix, normalize_adj, create_symm_matrix_from_vec, create_vec_from_symm_matrix
from gcn import GraphConvolution, GCNSynthetic

class GraphConvolutionPerturb(nn.Module):
	"""
	Similar to GraphConvolution except includes P_hat
	"""

	def __init__(self, in_features, out_features, bias=True):
		super(GraphConvolutionPerturb, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.weight = Parameter(torch.FloatTensor(in_features, out_features))
		if bias is not None:
			self.bias = Parameter(torch.FloatTensor(out_features))
		else:
			self.register_parameter('bias', None)


	def forward(self, input, adj):
		support = torch.mm(input, self.weight)
		output = torch.spmm(adj, support)
		if self.bias is not None:
			return output + self.bias
		else:
			return output

	def __repr__(self):
		return self.__class__.__name__ + ' (' \
		       + str(self.in_features) + ' -> ' \
		       + str(self.out_features) + ')'



class GCNSyntheticPerturb(nn.Module):
	"""
	3-layer GCN used in GNN Explainer synthetic tasks
	"""
	def __init__(self, nfeat, nhid, nout, nclass, adj, dropout, beta, edge_additions=False):
		super(GCNSyntheticPerturb, self).__init__()
		self.adj = adj
		self.nclass = nclass
		self.beta = beta
		self.num_nodes = self.adj.shape[0]
		self.edge_additions = edge_additions      # are edge additions included in perturbed matrix

		# P_hat needs to be symmetric ==> learn vector representing entries in upper/lower triangular matrix and use to populate P_hat later
		self.P_vec_size = int((self.num_nodes * self.num_nodes - self.num_nodes) / 2)  + self.num_nodes

		if self.edge_additions:
			self.P_vec = Parameter(torch.FloatTensor(torch.zeros(self.P_vec_size)))
		else:
			self.P_vec = Parameter(torch.FloatTensor(torch.ones(self.P_vec_size)))

		self.reset_parameters()

		self.gc1 = GraphConvolutionPerturb(nfeat, nhid)
		self.gc2 = GraphConvolutionPerturb(nhid, nhid)
		self.gc3 = GraphConvolution(nhid, nout)
		self.lin = nn.Linear(nhid + nhid + nout, nclass)
		self.dropout = dropout

	def reset_parameters(self, eps=10**-4):
		# Think more about how to initialize this
		with torch.no_grad():
			if self.edge_additions:
				adj_vec = create_vec_from_symm_matrix(self.adj, self.P_vec_size).numpy()
				for i in range(len(adj_vec)):
					if i < 1:
						adj_vec[i] = adj_vec[i] - eps
					else:
						adj_vec[i] = adj_vec[i] + eps
				torch.add(self.P_vec, torch.FloatTensor(adj_vec))       #self.P_vec is all 0s
			else:
				torch.sub(self.P_vec, eps)




	def forward(self, x, sub_adj):
		self.sub_adj = sub_adj
		# Same as normalize_adj in utils.py except includes P_hat in A_tilde
		self.P_hat_symm = create_symm_matrix_from_vec(self.P_vec, self.num_nodes)      # Ensure symmetry

		A_tilde = torch.FloatTensor(self.num_nodes, self.num_nodes)
		A_tilde.requires_grad = True

		if self.edge_additions:         # Learn new adj matrix directly
			A_tilde = F.sigmoid(self.P_hat_symm) + torch.eye(self.num_nodes)  # Use sigmoid to bound P_hat in [0,1]
		else:       # Learn P_hat that gets multiplied element-wise with adj -- only edge deletions
			A_tilde = F.sigmoid(self.P_hat_symm) * self.sub_adj + torch.eye(self.num_nodes)       # Use sigmoid to bound P_hat in [0,1]

		D_tilde = get_degree_matrix(A_tilde).detach()       # Don't need gradient of this
		# Raise to power -1/2, set all infs to 0s
		D_tilde_exp = D_tilde ** (-1 / 2)
		D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

		# Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
		norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)

		x1 = F.relu(self.gc1(x, norm_adj))
		x1 = F.dropout(x1, self.dropout, training=self.training)
		x2 = F.relu(self.gc2(x1, norm_adj))
		x2 = F.dropout(x2, self.dropout, training=self.training)
		x3 = self.gc3(x2, norm_adj)
		x = self.lin(torch.cat((x1, x2, x3), dim=1))
		return F.log_softmax(x, dim=1)


	def forward_prediction(self, x):
		# Same as forward but uses P instead of P_hat ==> non-differentiable
		# but needed for actual predictions

		self.P = (F.sigmoid(self.P_hat_symm) >= 0.5).float()      # threshold P_hat

		if self.edge_additions:
			A_tilde = self.P + torch.eye(self.num_nodes)
		else:
			A_tilde = self.P * self.adj + torch.eye(self.num_nodes)

		D_tilde = get_degree_matrix(A_tilde)
		# Raise to power -1/2, set all infs to 0s
		D_tilde_exp = D_tilde ** (-1 / 2)
		D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

		# Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
		norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)

		x1 = F.relu(self.gc1(x, norm_adj))
		x1 = F.dropout(x1, self.dropout, training=self.training)
		x2 = F.relu(self.gc2(x1, norm_adj))
		x2 = F.dropout(x2, self.dropout, training=self.training)
		x3 = self.gc3(x2, norm_adj)
		x = self.lin(torch.cat((x1, x2, x3), dim=1))
		return F.log_softmax(x, dim=1), self.P


	def  loss(self, output, y_pred_orig, y_pred_new_actual):
		pred_same = (y_pred_new_actual == y_pred_orig).float()

		# Need dim >=2 for F.nll_loss to work
		output = output.unsqueeze(0)
		y_pred_orig = y_pred_orig.unsqueeze(0)

		if self.edge_additions:
			cf_adj = self.P
		else:
			cf_adj = self.P * self.adj
		cf_adj.requires_grad = True  # Need to change this otherwise loss_graph_dist has no gradient

		# Want negative in front to maximize loss instead of minimizing it to find CFs
		loss_pred = - F.nll_loss(output, y_pred_orig)
		loss_graph_dist = sum(sum(abs(cf_adj - self.adj))) / 2      # Number of edges changed (symmetrical)

		# Zero-out loss_pred with pred_same if prediction flips
		loss_total = pred_same * loss_pred + self.beta * loss_graph_dist
		return loss_total, loss_pred, loss_graph_dist, cf_adj