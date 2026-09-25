import torch
from torch import nn
from torch.nn import functional as F
from peft import get_peft_model, LoraConfig
from .modules.alignment import HHAlignment
from .modules.encoder import ResMLPEncoder
from .modules.pathway import BoxMolecularOntology, PathwayGuidedDecoder
def encoder_lora(image_encoder_name:str, image_encoder, last_layer:int=3, is_lora_ffn:bool=False, lora_rank:int=8, lora_alpha:int=16, lora_dropout:float=0.1, logger=None):

    if last_layer > 0:

        if image_encoder_name == 'uni':
            length = range(24)
            lora_list = [f'blocks.{i}.attn.qkv'  for i in length[-last_layer:]]
            if is_lora_ffn:
                mlp_fc1_list = [f'blocks.{i}.mlp.fc1'  for i in length[-last_layer:]]
                mlp_fc2_list = [f'blocks.{i}.mlp.fc2'  for i in length[-last_layer:]]
            else:
                mlp_fc1_list = []
                mlp_fc2_list = []
        else:
            raise ValueError(f'image_encoder_name {image_encoder_name} is not support')
        lora_list = lora_list + mlp_fc1_list + mlp_fc2_list

        lora_config = LoraConfig(
            r=lora_rank,  
            lora_alpha=lora_alpha,  
            target_modules=lora_list,  
            lora_dropout=lora_dropout,  
            bias="none", 
        )

        peft_image_encoder = get_peft_model(image_encoder, lora_config)

        return peft_image_encoder
    else:
        for param in image_encoder.parameters():
            param.requires_grad = False
        return image_encoder

    
class HyBoxSTBase(nn.Module):
    def __init__(
        self,
        image_dim:int,
        gene_dim:int,
        emb_dim:int,
        num_outputs:int,
        mlp_ratio:float,
        image_dropout:float=0.1,
        gene_dropout:float=0,
        decoder_dropout:float=0.,

        entail_weight: float = 0.4,
        niche_project: bool = True,
        predict_norm: bool = False,
        alignment_beta: float = 0.2,


        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        image_encoder = None,
        logger=None,
        image_encoder_name:str = 'uni',
        last_layer:int = 3,
        is_lora_ffn:bool = False
        ):
        super().__init__()

        assert image_encoder_name in ['uni']
        self.alignment_beta = alignment_beta
        self.predict_norm = predict_norm


        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        self.last_layer = last_layer
        self.is_lora_ffn = is_lora_ffn

        self.image_encoder_name = image_encoder_name
        
        self.image_encoder = encoder_lora(
            image_encoder_name=image_encoder_name, image_encoder=image_encoder, 
            last_layer=last_layer, is_lora_ffn=is_lora_ffn, lora_rank=lora_rank, 
            lora_alpha=lora_alpha, lora_dropout=lora_dropout, logger=logger
        )

      
        self.image_projector = nn.Linear(
            in_features=image_dim,
            out_features=emb_dim
        )

        self.niche_image_projector = nn.Linear(
            in_features=image_dim,
            out_features=emb_dim
        )


        self.gene_encoder = nn.Linear(
            in_features=gene_dim,
            out_features=emb_dim
        )

        self.niche_gene_encoder = nn.Linear(
            in_features=gene_dim,
            out_features=emb_dim
        )
        self.alignment = HHAlignment(
            image_dim=emb_dim,
            gene_dim=emb_dim,
            embed_dim=emb_dim,
            mlp_ratio=mlp_ratio,
            image_dropout=image_dropout,
            gene_dropout=gene_dropout,
            entail_weight=entail_weight,
            niche_project=niche_project,
            predict_norm=predict_norm
        )

        self.gene_decoder = ResMLPEncoder(
            in_features=emb_dim * 2,
            hidden_size=emb_dim,
            mlp_ratio=mlp_ratio,
            drop=decoder_dropout,
            depth=2,
        )


        self.fc = nn.Linear(emb_dim, num_outputs)


    def forward(
            self,
            spot_img: torch.Tensor, niche_image: torch.Tensor,
            spot_gene_ebd: torch.Tensor, niche_gene_ebd: torch.Tensor,
            label:torch.Tensor
    ):

        image_emb = self.image_encoder(spot_img)
        niche_image_emb = self.image_encoder(niche_image)
        

        image_emb = self.image_projector(image_emb)

        niche_image_emb = self.niche_image_projector(niche_image_emb)

        if self.alignment_beta > 0:

            gene_emb = self.gene_encoder(spot_gene_ebd)

            niche_gene_emb = self.niche_gene_encoder(niche_gene_ebd)

            align_result = self.alignment(
                image_emb=image_emb,
                niche_image_emb=niche_image_emb,
                gene_emb=gene_emb,
                niche_gene_emb=niche_gene_emb
            )
            align_loss = align_result['loss']

            image_feats = align_result['emb']['image_feats']
            niche_image_feats = align_result['emb']['niche_image_feats']
            image_predict_emb = torch.cat([image_feats, niche_image_feats], dim=-1)

        else:
            align_result = None
            image_predict_emb = torch.cat([image_emb, niche_image_emb] , dim=-1)
          


        gene_pred = self.fc(self.gene_decoder(image_predict_emb))

        predict_loss = F.mse_loss(gene_pred, label)

        loss = predict_loss
        if self.alignment_beta > 0:
            loss = loss + self.alignment_beta * align_loss

        return {
            'loss': loss, 
            'logits': gene_pred, 
            'logging': {
                'predict_loss': predict_loss,
                'alignment_beta': self.alignment_beta,
                'align' : align_result
            }
        }


class HyBoxST(HyBoxSTBase):
    def __init__(
        self,
        gene_pathway_mask: torch.Tensor,
        pathway_box_dim: int = 128,
        pathway_loss_weight: float = 0.1,
        box_loss_weight: float = 0.01,
        prediction_mae_weight: float = 0.1,
        prediction_mean_weight: float = 0.05,
        prediction_std_weight: float = 0.02,
        prediction_corr_weight: float = 0.0,
        prediction_corr_min_label_std: float = 1e-4,
        prediction_corr_train_only: bool = True,
        high_expression_loss_weight: float = 0.0,
        direct_pathway_loss_weight: float = 0.05,
        use_query_box: bool = True,
        query_box_temperature: float = 0.1,
        min_query_box_size: float = 1e-3,
        query_mix_init: float = -1.38629436,
        use_gene_calibration: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        gene_pathway_mask = gene_pathway_mask.float()
        num_pathways, num_genes = gene_pathway_mask.shape
        if num_genes != kwargs["num_outputs"]:
            raise ValueError(
                f"gene_pathway_mask has {num_genes} genes, expected {kwargs['num_outputs']}."
        )

        self.pathway_loss_weight = pathway_loss_weight
        self.box_loss_weight = box_loss_weight
        self.prediction_mae_weight = prediction_mae_weight
        self.prediction_mean_weight = prediction_mean_weight
        self.prediction_std_weight = prediction_std_weight
        self.prediction_corr_weight = prediction_corr_weight
        self.prediction_corr_min_label_std = prediction_corr_min_label_std
        self.prediction_corr_train_only = prediction_corr_train_only
        self.high_expression_loss_weight = high_expression_loss_weight
        self.direct_pathway_loss_weight = direct_pathway_loss_weight
        self.pathway_guided_decoder = PathwayGuidedDecoder(
            in_features=kwargs["emb_dim"] * 2,
            hidden_size=kwargs["emb_dim"],
            num_pathways=num_pathways,
            num_genes=num_genes,
            gene_pathway_mask=gene_pathway_mask,
            box_dim=pathway_box_dim,
            mlp_ratio=kwargs["mlp_ratio"],
            drop=kwargs.get("decoder_dropout", 0.0),
            use_query_box=use_query_box,
            query_box_temperature=query_box_temperature,
            min_query_box_size=min_query_box_size,
            query_mix_init=query_mix_init,
            use_gene_calibration=use_gene_calibration,
        )
        self.box_ontology = BoxMolecularOntology(
            gene_pathway_mask=gene_pathway_mask,
            box_dim=pathway_box_dim,
        )

    def prediction_correlation_loss(self, gene_pred: torch.Tensor, label: torch.Tensor):
        pred = gene_pred.float()
        target = label.float()
        pred_centered = pred - pred.mean(dim=0, keepdim=True)
        target_centered = target - target.mean(dim=0, keepdim=True)
        pred_std = pred_centered.pow(2).mean(dim=0).clamp_min(1e-12).sqrt()
        target_std = target_centered.pow(2).mean(dim=0).clamp_min(1e-12).sqrt()
        valid_mask = target_std > float(self.prediction_corr_min_label_std)
        if int(valid_mask.sum().item()) == 0:
            zero = gene_pred.new_tensor(0.0)
            return zero, zero

        corr = (pred_centered * target_centered).mean(dim=0) / (pred_std * target_std).clamp_min(1e-12)
        corr = corr.clamp(min=-1.0, max=1.0)
        corr_loss = 1.0 - corr[valid_mask].mean()
        mean_corr = corr[valid_mask].detach().mean()
        return corr_loss.to(gene_pred.dtype), mean_corr.to(gene_pred.dtype)

    def prediction_loss(self, gene_pred: torch.Tensor, label: torch.Tensor):
        mse_loss = F.mse_loss(gene_pred, label)
        mae_loss = F.l1_loss(gene_pred, label)
        mean_loss = F.mse_loss(gene_pred.mean(dim=0), label.mean(dim=0))
        std_loss = F.mse_loss(
            gene_pred.std(dim=0, unbiased=False),
            label.std(dim=0, unbiased=False),
        )
        loss = mse_loss
        loss = loss + self.prediction_mae_weight * mae_loss
        loss = loss + self.prediction_mean_weight * mean_loss
        loss = loss + self.prediction_std_weight * std_loss
        corr_loss = gene_pred.new_tensor(0.0)
        corr_mean = gene_pred.new_tensor(0.0)
        if self.prediction_corr_weight > 0 and (self.training or not self.prediction_corr_train_only):
            corr_loss, corr_mean = self.prediction_correlation_loss(gene_pred, label)
            loss = loss + self.prediction_corr_weight * corr_loss
        high_expression_loss = gene_pred.new_tensor(0.0)
        if self.high_expression_loss_weight > 0:
            label_ref = label.detach()
            high_threshold = label_ref.mean(dim=0, keepdim=True) + label_ref.std(dim=0, unbiased=False, keepdim=True)
            high_mask = (label_ref > high_threshold).to(gene_pred.dtype)
            high_expression_loss = (torch.abs(gene_pred - label) * high_mask).sum()
            high_expression_loss = high_expression_loss / high_mask.sum().clamp_min(1.0)
            loss = loss + self.high_expression_loss_weight * high_expression_loss
        return {
            "loss": loss,
            "mse_loss": mse_loss,
            "mae_loss": mae_loss,
            "mean_loss": mean_loss,
            "std_loss": std_loss,
            "corr_loss": corr_loss,
            "corr_mean": corr_mean,
            "high_expression_loss": high_expression_loss,
        }

    def forward(
            self,
            spot_img: torch.Tensor, niche_image: torch.Tensor,
            spot_gene_ebd: torch.Tensor, niche_gene_ebd: torch.Tensor,
            label: torch.Tensor,
            spot_pathway_ebd: torch.Tensor = None,
            niche_pathway_ebd: torch.Tensor = None
    ):
        if spot_pathway_ebd is None:
            raise ValueError("HyBoxST requires spot_pathway_ebd from the dataset.")

        image_emb = self.image_encoder(spot_img)
        niche_image_emb = self.image_encoder(niche_image)

        image_emb = self.image_projector(image_emb)
        niche_image_emb = self.niche_image_projector(niche_image_emb)

        if self.alignment_beta > 0:
            gene_emb = self.gene_encoder(spot_gene_ebd)
            niche_gene_emb = self.niche_gene_encoder(niche_gene_ebd)

            align_result = self.alignment(
                image_emb=image_emb,
                niche_image_emb=niche_image_emb,
                gene_emb=gene_emb,
                niche_gene_emb=niche_gene_emb
            )
            align_loss = align_result['loss']

            image_feats = align_result['emb']['image_feats']
            niche_image_feats = align_result['emb']['niche_image_feats']
            image_predict_emb = torch.cat([image_feats, niche_image_feats], dim=-1)
        else:
            align_result = None
            align_loss = None
            image_predict_emb = torch.cat([image_emb, niche_image_emb], dim=-1)

        decoder_result = self.pathway_guided_decoder(
            image_predict_emb,
            box_ontology=self.box_ontology,
        )
        gene_pred = decoder_result["gene_pred"]
        pathway_pred = decoder_result["pathway_pred"]

        predict_result = self.prediction_loss(gene_pred, label)
        predict_loss = predict_result["loss"]
        pathway_loss = F.mse_loss(pathway_pred, spot_pathway_ebd)
        direct_pathway_loss = F.mse_loss(
            decoder_result["direct_pathway_pred"],
            spot_pathway_ebd,
        )
        box_result = self.box_ontology()
        box_loss = box_result["loss"]

        loss = predict_loss
        if self.alignment_beta > 0:
            loss = loss + self.alignment_beta * align_loss
        loss = loss + self.pathway_loss_weight * pathway_loss
        loss = loss + self.direct_pathway_loss_weight * direct_pathway_loss
        loss = loss + self.box_loss_weight * box_loss

        return {
            'loss': loss,
            'logits': gene_pred,
            'logging': {
                'predict_loss': predict_loss,
                'predict_mse_loss': predict_result["mse_loss"],
                'predict_mae_loss': predict_result["mae_loss"],
                'predict_mean_loss': predict_result["mean_loss"],
                'predict_std_loss': predict_result["std_loss"],
                'predict_corr_loss': predict_result["corr_loss"],
                'predict_corr_mean': predict_result["corr_mean"],
                'predict_high_expression_loss': predict_result["high_expression_loss"],
                'pathway_loss': pathway_loss,
                'direct_pathway_loss': direct_pathway_loss,
                'box_loss': box_loss,
                'box_positive_loss': box_result["positive_loss"],
                'box_negative_loss': box_result["negative_loss"],
                'alignment_beta': self.alignment_beta,
                'pathway_loss_weight': self.pathway_loss_weight,
                'direct_pathway_loss_weight': self.direct_pathway_loss_weight,
                'box_loss_weight': self.box_loss_weight,
                'prediction_mae_weight': self.prediction_mae_weight,
                'prediction_mean_weight': self.prediction_mean_weight,
                'prediction_std_weight': self.prediction_std_weight,
                'prediction_corr_weight': self.prediction_corr_weight,
                'prediction_corr_min_label_std': self.prediction_corr_min_label_std,
                'prediction_corr_train_only': self.prediction_corr_train_only,
                'high_expression_loss_weight': self.high_expression_loss_weight,
                'align': align_result,
                'pathway_pred': pathway_pred,
                'direct_pathway_pred': decoder_result["direct_pathway_pred"],
                'query_overlap': decoder_result["query_overlap"],
                'query_pathway_residual': decoder_result["query_pathway_residual"],
                'query_mix': decoder_result["query_mix"],
                'query_lower': decoder_result["query_lower"],
                'query_upper': decoder_result["query_upper"],
                'raw_gene_pred': decoder_result["raw_gene_pred"],
                'pathway_gene_pred': decoder_result["pathway_gene_pred"],
                'residual_pred': decoder_result["residual_pred"],
                'gene_gate': decoder_result["gene_gate"],
            }
        }
