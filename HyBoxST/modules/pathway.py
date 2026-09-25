import torch
from torch import nn
from torch.nn import functional as F

from .encoder import ResMLPEncoder


class BoxMolecularOntology(nn.Module):
    def __init__(
        self,
        gene_pathway_mask: torch.Tensor,
        box_dim: int = 128,
        margin: float = 1.0,
        negative_weight: float = 0.1,
        min_box_size: float = 1e-3,
    ):
        super().__init__()
        if gene_pathway_mask.ndim != 2:
            raise ValueError("gene_pathway_mask must have shape (num_pathways, num_genes).")

        self.register_buffer("gene_pathway_mask", gene_pathway_mask.float())
        num_pathways, num_genes = gene_pathway_mask.shape
        self.pathway_center = nn.Parameter(torch.empty(num_pathways, box_dim))
        self.pathway_log_size = nn.Parameter(torch.empty(num_pathways, box_dim))
        self.gene_point = nn.Parameter(torch.empty(num_genes, box_dim))
        self.margin = margin
        self.negative_weight = negative_weight
        self.min_box_size = min_box_size
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.pathway_center, std=0.02)
        nn.init.normal_(self.pathway_log_size, mean=-2.0, std=0.02)
        nn.init.normal_(self.gene_point, std=0.02)

    def box_bounds(self):
        half_size = F.softplus(self.pathway_log_size) + self.min_box_size
        lower = self.pathway_center - half_size
        upper = self.pathway_center + half_size
        return lower, upper

    def point_to_box_distance(self):
        lower, upper = self.box_bounds()
        gene = self.gene_point.unsqueeze(0)
        lower = lower.unsqueeze(1)
        upper = upper.unsqueeze(1)
        outside = F.relu(lower - gene) + F.relu(gene - upper)
        return torch.linalg.vector_norm(outside, dim=-1)

    def soft_overlap_with_query(
        self,
        query_lower: torch.Tensor,
        query_upper: torch.Tensor,
        temperature: float = 0.1,
        eps: float = 1e-6,
    ):
        pathway_lower, pathway_upper = self.box_bounds()
        query_lower = query_lower.unsqueeze(1)
        query_upper = query_upper.unsqueeze(1)
        pathway_lower = pathway_lower.unsqueeze(0)
        pathway_upper = pathway_upper.unsqueeze(0)

        inter_lower = torch.maximum(query_lower, pathway_lower)
        inter_upper = torch.minimum(query_upper, pathway_upper)
        inter_width = temperature * F.softplus((inter_upper - inter_lower) / temperature)
        query_width = (query_upper - query_lower).clamp_min(eps)

        # Geometric-mean overlap avoids high-dimensional volume underflow.
        log_overlap = torch.log(inter_width + eps) - torch.log(query_width + eps)
        return torch.exp(log_overlap.mean(dim=-1).clamp(min=-30.0, max=0.0))

    def forward(self):
        distances = self.point_to_box_distance()
        positive_mask = self.gene_pathway_mask.bool()
        negative_mask = ~positive_mask

        positive_loss = distances[positive_mask].mean()
        negative_loss = F.relu(self.margin - distances[negative_mask]).mean()
        loss = positive_loss + self.negative_weight * negative_loss
        return {
            "loss": loss,
            "positive_loss": positive_loss,
            "negative_loss": negative_loss,
        }


class PathwayGuidedDecoder(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        num_pathways: int,
        num_genes: int,
        gene_pathway_mask: torch.Tensor,
        box_dim: int = None,
        mlp_ratio: float = 2.0,
        drop: float = 0.0,
        use_query_box: bool = True,
        query_box_temperature: float = 0.1,
        min_query_box_size: float = 1e-3,
        query_mix_init: float = -1.38629436,
        use_gene_calibration: bool = True,
    ):
        super().__init__()
        if gene_pathway_mask.shape != (num_pathways, num_genes):
            raise ValueError("gene_pathway_mask shape does not match decoder dimensions.")

        self.register_buffer("gene_pathway_mask", gene_pathway_mask.float())
        covered_gene = (gene_pathway_mask.sum(dim=0) > 0).float()
        self.register_buffer("covered_gene", covered_gene)

        self.pathway_decoder = ResMLPEncoder(
            in_features=in_features,
            hidden_size=hidden_size,
            mlp_ratio=mlp_ratio,
            drop=drop,
            depth=2,
        )
        self.pathway_head = nn.Linear(hidden_size, num_pathways)

        self.residual_decoder = ResMLPEncoder(
            in_features=in_features,
            hidden_size=hidden_size,
            mlp_ratio=mlp_ratio,
            drop=drop,
            depth=2,
        )
        self.residual_head = nn.Linear(hidden_size, num_genes)

        self.pathway_gene_logit = nn.Parameter(torch.zeros(num_pathways, num_genes))
        self.gene_gate_logit = nn.Parameter(torch.zeros(num_genes))
        self.use_query_box = use_query_box and box_dim is not None
        self.query_box_temperature = query_box_temperature
        self.min_query_box_size = min_query_box_size
        self.use_gene_calibration = use_gene_calibration

        if self.use_query_box:
            self.query_box_decoder = ResMLPEncoder(
                in_features=in_features,
                hidden_size=hidden_size,
                mlp_ratio=mlp_ratio,
                drop=drop,
                depth=2,
            )
            self.query_box_head = nn.Linear(hidden_size, box_dim * 2)
            self.overlap_scale = nn.Parameter(torch.zeros(num_pathways))
            self.overlap_bias = nn.Parameter(torch.zeros(num_pathways))
            self.pathway_query_mix_logit = nn.Parameter(torch.tensor(float(query_mix_init)))

        if self.use_gene_calibration:
            self.gene_scale_logit = nn.Parameter(torch.full((num_genes,), 0.54132485))
            self.gene_bias = nn.Parameter(torch.zeros(num_genes))

    def masked_pathway_gene_weight(self):
        weights = F.softplus(self.pathway_gene_logit) * self.gene_pathway_mask
        return weights / weights.sum(dim=0, keepdim=True).clamp_min(1.0)

    def _query_box(self, image_predict_emb: torch.Tensor):
        query_params = self.query_box_head(self.query_box_decoder(image_predict_emb))
        query_center, query_log_size = query_params.chunk(2, dim=-1)
        query_half_size = F.softplus(query_log_size) + self.min_query_box_size
        return query_center - query_half_size, query_center + query_half_size

    def forward(self, image_predict_emb: torch.Tensor, box_ontology: BoxMolecularOntology = None):
        direct_pathway_pred = self.pathway_head(self.pathway_decoder(image_predict_emb))
        pathway_pred = direct_pathway_pred
        query_overlap = None
        query_pathway_residual = None
        query_mix = None
        query_lower = None
        query_upper = None

        if self.use_query_box:
            if box_ontology is None:
                raise ValueError("PathwayGuidedDecoder with query box requires box_ontology.")
            query_lower, query_upper = self._query_box(image_predict_emb)
            query_overlap = box_ontology.soft_overlap_with_query(
                query_lower=query_lower,
                query_upper=query_upper,
                temperature=self.query_box_temperature,
            )
            query_pathway_residual = query_overlap * self.overlap_scale + self.overlap_bias
            query_mix = torch.sigmoid(self.pathway_query_mix_logit)
            pathway_pred = direct_pathway_pred + query_mix * query_pathway_residual

        residual_pred = self.residual_head(self.residual_decoder(image_predict_emb))
        pathway_gene_weight = self.masked_pathway_gene_weight()
        pathway_gene_pred = pathway_pred @ pathway_gene_weight

        gate = torch.sigmoid(self.gene_gate_logit) * self.covered_gene
        raw_gene_pred = gate * pathway_gene_pred + (1.0 - gate) * residual_pred
        if self.use_gene_calibration:
            gene_pred = raw_gene_pred * F.softplus(self.gene_scale_logit) + self.gene_bias
        else:
            gene_pred = raw_gene_pred

        return {
            "gene_pred": gene_pred,
            "raw_gene_pred": raw_gene_pred,
            "pathway_pred": pathway_pred,
            "direct_pathway_pred": direct_pathway_pred,
            "query_overlap": query_overlap,
            "query_pathway_residual": query_pathway_residual,
            "query_mix": query_mix,
            "query_lower": query_lower,
            "query_upper": query_upper,
            "pathway_gene_pred": pathway_gene_pred,
            "residual_pred": residual_pred,
            "gene_gate": gate,
        }
