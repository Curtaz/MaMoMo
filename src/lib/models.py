import dgl
import dgl.nn as gnn
import torch
import torch.nn.functional as F
from matgl.config import DEFAULT_ELEMENT_TYPES
from matgl.layers import (MLP, BondExpansion, MEGNetBlock, SoftExponential,
                          SoftPlus2,EdgeSet2Set,EmbeddingBlock,
                          GatedMLP,EdgeSet2Set)
from dgl.nn import Set2Set

from torch import nn

class MyGNN(nn.Module):
    def __init__(self, 
                 num_node_types, 
                 embed_dim,
                 conv_dims, 
                 ):
        
        super(MyGNN, self).__init__()
        self.embedding = nn.Embedding(num_node_types,embed_dim)
        self.convs = nn.Sequential()
        last_hidden_dim = embed_dim

        for dim in conv_dims:
            self.convs.append(gnn.SAGEConv(last_hidden_dim,dim,aggregator_type='mean'))
            last_hidden_dim = dim

    def forward(self, g, in_feat):
        h = self.embedding(in_feat)
        for layer in self.convs: 
            h = layer(g, h)            
            h = F.relu(h)
        
        return h

class MySAGEClassifier(nn.Module):
    def __init__(self, 
                 num_node_types, 
                 embed_dim,
                 conv_dims, 
                 ):
        
        super(MySAGEClassifier, self).__init__()
        self.gnn = MyGNN(
                 num_node_types, 
                 embed_dim,
                 conv_dims,
                 )
        
        self.predictor =  gnn.SAGEConv(conv_dims[-1],num_node_types,aggregator_type='mean',dropout=0.3)
        
    def forward(self, g, in_feat):
        h = self.gnn(g,in_feat)
        h = self.predictor(g,h)
        h = F.softmax(h,dim=1)
        return h

"""
GAT 
Graph Attention Network, Attention convolution as described in  "Graph Attention Networks ()"
"""

class MyGAT(nn.Module):
    def __init__(self, 
                 num_node_types, 
                 embed_dim,
                 attn_dims, 
                 attn_heads,
                 activation = F.elu, 
                 ):
        
        super(MyGAT, self).__init__()
        self.embedding = nn.Embedding(num_node_types,embed_dim)
        self.convs = nn.Sequential()
        self.activation = activation
        
        assert len(attn_dims)==len(attn_heads)
        last_hidden_dim = embed_dim
        for i,dim in enumerate(attn_dims):
            self.convs.append(gnn.GATv2Conv(last_hidden_dim,dim,attn_heads[i]))
            last_hidden_dim = dim*attn_heads[i]

    def forward(self, g, in_feat):
        h = self.embedding(in_feat)
        for layer in self.convs: 
            h = layer(g,h).view(g.num_nodes(),-1)         
            h = self.activation(h)
        
        return h
    
class MyGATClassifier(nn.Module):
    def __init__(self, 
                num_node_types, 
                embed_dim,
                attn_dims, 
                attn_heads,
                classifier_attn_heads,
                ):
        
        super(MyGATClassifier, self).__init__()
        self.gat = MyGAT(num_node_types,embed_dim, attn_dims,attn_heads)
        self.classifier = gnn.GATConv(attn_dims[-1]*attn_heads[-1],num_node_types,classifier_attn_heads)
        

    def forward(self, g, in_feat):
        h = self.gat(g,in_feat)
        h = self.classifier(g,h)
        h = h.mean(dim=1)
        return F.softmax(h,dim=1)



"""
EGAT
Graph Attention Network with edge features
"""
class MyEGAT(nn.Module):
    def __init__(self, 
                 num_node_types, 
                 node_embed_dim,
                 edge_embed_dim,
                 edge_dims,
                 attn_dims, 
                 attn_heads,
                 activation = F.elu, 
                 ):
        
        super(MyEGAT, self).__init__()
        self.embedding = nn.Embedding(num_node_types,node_embed_dim)
        self.convs = nn.Sequential()
        self.activation = activation
        
        assert len(attn_dims)==len(attn_heads)
        last_hidden_dim = node_embed_dim
        last_edge_dim = edge_embed_dim
    
        for i,out_dim in enumerate(attn_dims):
            self.convs.append(gnn.EGATConv(last_hidden_dim,last_edge_dim,out_dim,edge_dims[i],attn_heads[i]))
            last_hidden_dim = out_dim*attn_heads[i]
            last_edge_dim = edge_dims[i]*attn_heads[i]

    def forward(self, g, nfeats, efeats):
        nfeats = self.embedding(nfeats)
        for layer in self.convs: 
            nfeats,efeats = layer(g,nfeats,efeats) 
            nfeats = self.activation(nfeats.view(g.num_nodes(),-1))
            efeats = self.activation(efeats.view(g.num_edges(),-1))    

        return nfeats,efeats
    
class MyEGATClassifier(nn.Module):
    def __init__(self, 
                num_node_types, 
                node_embed_dim,
                edge_embed_dim,
                edge_dims,
                attn_dims, 
                attn_heads,
                classifier_attn_heads,
                ):
        
        super(MyEGATClassifier, self).__init__()
        self.gat = MyEGAT(num_node_types,node_embed_dim,edge_embed_dim,edge_dims,attn_dims,attn_heads)
        self.classifier = gnn.EGATConv(attn_dims[-1]*attn_heads[-1],edge_dims[-1]*attn_heads[-1],num_node_types,1,classifier_attn_heads)
        

    def forward(self, g, nfeats,efeats):
        node_feats,edge_feats = self.gat(g,nfeats,efeats)
        out_feats,_ = self.classifier(g,node_feats,edge_feats)
        out_feats = out_feats.mean(dim=1)
        return F.softmax(out_feats,dim=1)

class MyEGATRegressor(nn.Module):
    def __init__(self, 
                num_node_types, 
                node_embed_dim,
                edge_embed_dim,
                edge_dims,
                attn_dims, 
                attn_heads,
                niters_set2set,
                nlayers_set2set,
                hidden_layer_sizes_output,
                is_classification = False
                ):
        
        super(MyEGATRegressor, self).__init__()
        self.gat = MyEGAT(num_node_types,node_embed_dim,edge_embed_dim,edge_dims,attn_dims,attn_heads)
        s2s_kwargs = {"n_iters": niters_set2set, "n_layers": nlayers_set2set}

        edges_out_dim = edge_dims[-1]*attn_heads[-1]
        nodes_out_dim = attn_dims[-1]*attn_heads[-1]
        self.node_s2s = gnn.Set2Set(nodes_out_dim, **s2s_kwargs)
        self.edge_s2s = EdgeSet2Set(edges_out_dim,n_iters = 1,n_layers = 2)
        self.output_proj = GatedMLP(
            # S2S cats q_star to output producing double the dim
            2*nodes_out_dim + 2*edges_out_dim, 
            hidden_layer_sizes_output,
            activate_last=False,
        )

        self.is_classification = is_classification                

    def forward(self, g, nfeats,efeats):
        node_feat,edge_feat = self.gat(g,nfeats,efeats)
        node_vec = self.node_s2s(g, node_feat)
        edge_vec = self.edge_s2s(g, edge_feat)

        node_vec = torch.squeeze(node_vec)
        edge_vec = torch.squeeze(edge_vec)
        vec = torch.hstack([node_vec, edge_vec])

        out = self.output_proj(vec)
        if self.is_classification:
            out = F.softmax(out,dim=1)
        return out
    
class AllInOneEGAT(nn.Module):
    def __init__(self, 
                num_node_types, 
                node_embed_dim,
                edge_embed_dim,
                edge_dims,
                attn_dims, 
                attn_heads,
                classifier_attn_heads,
                niters_set2set,
                nlayers_set2set,
                hidden_layer_sizes_output,
                is_graph_classification = False
                ):
        
        super(AllInOneEGAT, self).__init__()
        self.gat = MyEGAT(num_node_types,node_embed_dim,edge_embed_dim,edge_dims,attn_dims,attn_heads)
        s2s_kwargs = {"n_iters": niters_set2set, "n_layers": nlayers_set2set}

        edges_out_dim = edge_dims[-1]*attn_heads[-1]
        nodes_out_dim = attn_dims[-1]*attn_heads[-1]
        self.node_s2s = gnn.Set2Set(nodes_out_dim, **s2s_kwargs)
        self.edge_s2s = EdgeSet2Set(edges_out_dim,n_iters = 1,n_layers = 2)
        self.regressor = GatedMLP(
            # S2S cats q_star to output producing double the dim
            2*nodes_out_dim + 2*edges_out_dim, 
            hidden_layer_sizes_output,
            activate_last=False,
        )
    
        self.classifier = gnn.EGATConv(nodes_out_dim,edges_out_dim,num_node_types,1,classifier_attn_heads)

        self.is_graph_classification = is_graph_classification                

    def forward(self, g, nfeats,efeats):
        node_feat,edge_feat = self.gat(g,nfeats,efeats)
        
        # Graph level prediction
        node_vec = self.node_s2s(g, node_feat)
        edge_vec = self.edge_s2s(g, edge_feat)
        node_vec = torch.squeeze(node_vec)
        edge_vec = torch.squeeze(edge_vec)
        vec = torch.hstack([node_vec, edge_vec])
        g_out = self.regressor(vec)
        if self.is_graph_classification:
            g_out = F.softmax(g_out,dim=1)

        # Node and Edge level predictions
        n_out,e_out = self.classifier(g,node_feat,edge_feat)
        n_out = n_out.mean(dim=1)
        e_out = e_out.mean(dim=1)

        return {'graph':g_out,
                'node':n_out,
                'edge':e_out,
                }