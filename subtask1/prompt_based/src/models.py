#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-FileCopyrightText: Copyright © <2022> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
#
# SPDX-License-Identifier: MIT-License

"""Custom models for few-shot learning specific operations."""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import (
    BertModel,
    BertOnlyMLMHead,
    BertPreTrainedModel,
)
from transformers.models.deberta.modeling_deberta import (
    ContextPooler,
    DebertaModel,
    DebertaOnlyMLMHead,
    DebertaPreTrainedModel,
    StableDropout,
)
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    ContextPooler,
    DebertaV2Model,
    DebertaV2OnlyMLMHead,
    DebertaV2PreTrainedModel,
    StableDropout,
)
from transformers.models.roberta.modeling_roberta import (
    RobertaClassificationHead,
    RobertaLMHead,
    RobertaModel,
)

from .loss import ContrastiveLoss, SymKlCriterion, stable_kl

logger = logging.getLogger(__name__)


def generate_noise(embed, epsilon=1e-5):
    noise = embed.data.new(embed.size()).normal_(0, 1) * epsilon
    noise.detach()
    noise.requires_grad_()
    return noise


def norm_grad(grad, eff_grad=None, sentence_level=False, norm_p="max", epsilon=1e-5):
    if norm_p == "l2":
        if sentence_level:
            direction = grad / (torch.norm(grad, dim=(-2, -1), keepdim=True) + epsilon)
        else:
            direction = grad / (torch.norm(grad, dim=-1, keepdim=True) + epsilon)
    elif norm_p == "l1":
        direction = grad.sign()
    else:
        if sentence_level:
            direction = grad / (grad.abs().max((-2, -1), keepdim=True)[0] + epsilon)
        else:
            direction = grad / (grad.abs().max(-1, keepdim=True)[0] + epsilon)
            eff_direction = eff_grad / (grad.abs().max(-1, keepdim=True)[0] + epsilon)
    return direction, eff_direction


def resize_token_type_embeddings(model, new_num_types: int, random_segment: bool):
    """
    Resize the segment (token type) embeddings for BERT
    """
    if hasattr(model, "bert"):
        old_token_type_embeddings = model.bert.embeddings.token_type_embeddings
    else:
        raise NotImplementedError
    new_token_type_embeddings = nn.Embedding(
        new_num_types, old_token_type_embeddings.weight.size(1)
    )
    if not random_segment:
        new_token_type_embeddings.weight.data[
            : old_token_type_embeddings.weight.size(0)
        ] = old_token_type_embeddings.weight.data

    model.config.type_vocab_size = new_num_types
    if hasattr(model, "bert"):
        model.bert.embeddings.token_type_embeddings = new_token_type_embeddings
    else:
        raise NotImplementedError


class BertForPromptFinetuning(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None

        # For label search.
        self.return_full_softmax = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        mask_pos=None,
        labels=None,
    ):
        batch_size = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # Encode everything
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

        # Get <mask> token representation
        sequence_output, pooled_output = outputs[:2]
        sequence_mask_output = sequence_output[
            torch.arange(sequence_output.size(0)), mask_pos
        ]

        # Logits over vocabulary tokens
        prediction_mask_scores = self.cls(sequence_mask_output)

        # Exit early and only return mask logits.
        if self.return_full_softmax:
            if labels is not None:
                return (
                    torch.zeros(1, out=prediction_mask_scores.new()),
                    prediction_mask_scores,
                )
            return prediction_mask_scores

        # Return logits for each label
        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(
                prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1)
            )
        logits = torch.cat(logits, -1)

        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits)  # Log prob of right polarity

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack(
                    [
                        1 - (labels.view(-1) - self.lb) / (self.ub - self.lb),
                        (labels.view(-1) - self.lb) / (self.ub - self.lb),
                    ],
                    -1,
                )
                loss = loss_fct(logits.view(-1, 2), labels)
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (
                torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,
            )
        return ((loss,) + output) if loss is not None else output


class RobertaForPromptFinetuning(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)
        self.lm_head = RobertaLMHead(config)
        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None

        # For auto label search.
        self.return_full_softmax = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mask_pos=None,
        labels=None,
    ):
        batch_size = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # Encode everything
        outputs = self.roberta(input_ids, attention_mask=attention_mask)

        # Get <mask> token representation
        sequence_output, pooled_output = outputs[:2]
        sequence_mask_output = sequence_output[
            torch.arange(sequence_output.size(0)), mask_pos
        ]

        # Logits over vocabulary tokens
        prediction_mask_scores = self.lm_head(sequence_mask_output)

        # Exit early and only return mask logits.
        if self.return_full_softmax:
            if labels is not None:
                return (
                    torch.zeros(1, out=prediction_mask_scores.new()),
                    prediction_mask_scores,
                )
            return prediction_mask_scores

        # Return logits for each label
        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(
                prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1)
            )
        logits = torch.cat(logits, -1)

        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits)  # Log prob of right polarity

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack(
                    [
                        1 - (labels.view(-1) - self.lb) / (self.ub - self.lb),
                        (labels.view(-1) - self.lb) / (self.ub - self.lb),
                    ],
                    -1,
                )
                loss = loss_fct(logits.view(-1, 2), labels)
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (
                torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,
            )
        return ((loss,) + output) if loss is not None else output


class DebertaForPromptFinetuning(DebertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        # self.deberta = DebertaV2Model(config)

        self.deberta = DebertaModel(config)
        self.cls = DebertaOnlyMLMHead(config)

        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim

        self.classifier = torch.nn.Linear(output_dim, self.num_labels)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out

        self.dropout = StableDropout(drop_out)

        classification_list = [self.pooler, self.dropout, self.classifier]

        self.classifier = nn.Sequential(*classification_list)
        # self.cls = DebertaV2OnlyMLMHead(config)

        self.map = nn.Linear(config.hidden_size, config.hidden_size)
        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None
        self.K = 1
        self.step_size = 1e-5
        self.adv_lc = SymKlCriterion()
        self.contra_lc = ContrastiveLoss()
        # import pdb
        # pdb.set_trace()
        # self.step_size=config.step_size

        # For auto label search.
        self.return_full_softmax = None

    def get_constrast_loss(
        self,
        input_ids=None,
        attention_mask=None,
        mask_pos=None,
        labels=None,
        inputs_embeds=None,
    ):

        self.cos = nn.CosineSimilarity(dim=-1)

        _, sequence_mask_output_1 = self.encode(
            input_ids, attention_mask, mask_pos, inputs_embeds
        )
        _, sequence_mask_output_2 = self.encode(
            input_ids, attention_mask, mask_pos, inputs_embeds
        )

        sequence_mask_output_1 = self.lm_head.dense(sequence_mask_output_1)
        sequence_mask_output_2 = self.lm_head.dense(sequence_mask_output_2)
        # input_args = [input_ids, attention_mask, mask_pos, labels, None, 1]
        # embed = self.forward(*input_args)
        #
        # vat_args = [input_ids, attention_mask, mask_pos, labels, embed, 2]
        #
        # adv_logits, outputs = self.forward(*vat_args)
        #
        # logit_mask = F.softmax(logits, dim=-1)[torch.arange(adv_logits.size(0)), labels] > 0.7
        #
        # outputs = outputs[logit_mask]
        # seq_outputs = sequence_mask_output[logit_mask]
        # new_label = labels[logit_mask]
        # #
        # #
        # rand_perm = torch.randperm(outputs.size(0))
        # rand_outputs = outputs[rand_perm, :]
        # rand_label = new_label[rand_perm]
        # pair_label = (new_label == rand_label).long()
        #
        # seq_outputs = self.map(seq_outputs)
        # rand_outputs = self.map(rand_outputs)

        pair_labels = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()

        # import  pdb
        # pdb.set_trace()

        contra_loss = self.contra_lc(
            sequence_mask_output_1.unsqueeze(1),
            sequence_mask_output_2.unsqueeze(0),
            pair_labels,
        )

        if torch.isnan(contra_loss):
            return 0

        return contra_loss

    def get_adv_loss(
        self,
        input_ids=None,
        attention_mask=None,
        mask_pos=None,
        labels=None,
        inputs_embeds=None,
    ):

        logits, sequence_mask_output = self.encode(
            input_ids, attention_mask, mask_pos, inputs_embeds
        )

        input_args = [input_ids, attention_mask, mask_pos, labels, None, 1]
        embed = self.forward(*input_args)
        noise = generate_noise(embed)

        for step in range(0, self.K):
            vat_args = [input_ids, attention_mask, mask_pos, labels, embed + noise, 2]
            adv_logits, _ = self.forward(*vat_args)
            adv_loss = stable_kl(adv_logits, logits.detach(), reduce=False)
            try:
                (delta_grad,) = torch.autograd.grad(
                    adv_loss, noise, only_inputs=True, retain_graph=False
                )
            except:
                import pdb

                pdb.set_trace()

            norm = delta_grad.norm()
            if torch.isnan(norm) or torch.isinf(norm):
                return 0
            eff_delta_grad = delta_grad * self.step_size
            delta_grad, eff_noise = norm_grad(delta_grad, eff_grad=eff_delta_grad)
            noise = noise + delta_grad * self.step_size
            # noise, eff_noise = self._norm_grad(delta_grad, eff_grad=eff_delta_grad, sentence_level=self.norm_level)
            noise = noise.detach()
            noise.requires_grad_()

        vat_args = [input_ids, attention_mask, mask_pos, labels, embed + noise, 2]

        adv_logits, sequence_mask_output = self.forward(*vat_args)
        # ori_args = model(*ori_args)
        # aug_args = [input_ids, token_type_ids, attention_mask, premise_mask, hyp_mask, task_id, 2, (embed + native_noise).detach()]

        adv_loss = self.adv_lc(adv_logits, logits)
        return adv_loss

    def embed_encode(self, input_ids):
        embedding_output = self.deberta.embeddings.word_embeddings(input_ids)
        return embedding_output

    def encode(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        mask_pos=None,
        inputs_embeds=None,
        return_full_softmax=False,
    ):
        batch_size = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # Encode everything
        if inputs_embeds is None:
            outputs = self.deberta(
                input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
            )
        else:
            outputs = self.deberta(
                None,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
            )

        # Get <mask> token representation
        sequence_output = outputs[0]
        sequence_mask_output = sequence_output[
            torch.arange(sequence_output.size(0)), mask_pos
        ]

        # Logits over vocabulary tokens
        prediction_mask_scores = self.cls(sequence_mask_output)

        # sequence_mask_output = self.lm_head.dense(sequence_mask_output)

        # Exit early and only return mask logits.
        if return_full_softmax:
            return prediction_mask_scores

        # Return logits for each label
        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(
                prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1)
            )
        logits = torch.cat(logits, -1)

        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits)  # Log prob of right polarity

        # if self.model_args.hybrid == 1:
        #     cls_logits = self.classifier(sequence_output)
        #     return (logits, cls_logits), sequence_mask_output

        return logits, sequence_mask_output

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        mask_pos=None,
        labels=None,
        inputs_embeds=None,
        fwd_type=0,
        block_flag=None,
    ):

        if fwd_type == 2:
            assert inputs_embeds is not None
            return self.encode(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                mask_pos=mask_pos,
                inputs_embeds=inputs_embeds,
            )

        elif fwd_type == 1:
            return self.embed_encode(input_ids)

        if self.data_args.continuous_prompt == 1 and block_flag is not None:
            inputs_embeds = self.generate_continuous_prompt_inputs(
                input_ids, block_flag
            )

        logits, sequence_mask_output = self.encode(
            input_ids, attention_mask, token_type_ids, mask_pos, inputs_embeds
        )

        # if self.model_args.hybrid == 1:
        #     logits = logits[0]
        #     cls_logits = logits[1]

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack(
                    [
                        1 - (labels.view(-1) - self.lb) / (self.ub - self.lb),
                        (labels.view(-1) - self.lb) / (self.ub - self.lb),
                    ],
                    -1,
                )
                loss = loss_fct(logits.view(-1, 2), labels)
            else:

                if labels.shape == logits.shape:
                    loss = F.kl_div(
                        F.log_softmax(logits, dim=-1, dtype=torch.float32),
                        labels,
                        reduction="batchmean",
                    )
                else:
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (
                torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,
            )

        return ((loss,) + output) if loss is not None else output


class Debertav2ForPromptFinetuning(DebertaV2PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.deberta = DebertaV2Model(config)
        self.cls = DebertaV2OnlyMLMHead(config)

        # self.deberta = DebertaModel(config)
        # self.cls = DebertaOnlyMLMHead(config)

        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim

        self.classifier = torch.nn.Linear(output_dim, self.num_labels)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out

        self.dropout = StableDropout(drop_out)

        classification_list = [self.pooler, self.dropout, self.classifier]

        self.classifier = nn.Sequential(*classification_list)
        # self.cls = DebertaV2OnlyMLMHead(config)

        self.map = nn.Linear(config.hidden_size, config.hidden_size)
        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None
        self.K = 1
        self.step_size = 1e-5
        self.adv_lc = SymKlCriterion()
        self.contra_lc = ContrastiveLoss()
        # import pdb
        # pdb.set_trace()
        # self.step_size=config.step_size

        # For auto label search.
        self.return_full_softmax = None

    def get_constrast_loss(
        self,
        input_ids=None,
        attention_mask=None,
        mask_pos=None,
        labels=None,
        inputs_embeds=None,
    ):

        self.cos = nn.CosineSimilarity(dim=-1)

        _, sequence_mask_output_1 = self.encode(
            input_ids, attention_mask, mask_pos, inputs_embeds
        )
        _, sequence_mask_output_2 = self.encode(
            input_ids, attention_mask, mask_pos, inputs_embeds
        )

        sequence_mask_output_1 = self.lm_head.dense(sequence_mask_output_1)
        sequence_mask_output_2 = self.lm_head.dense(sequence_mask_output_2)
        # input_args = [input_ids, attention_mask, mask_pos, labels, None, 1]
        # embed = self.forward(*input_args)
        #
        # vat_args = [input_ids, attention_mask, mask_pos, labels, embed, 2]
        #
        # adv_logits, outputs = self.forward(*vat_args)
        #
        # logit_mask = F.softmax(logits, dim=-1)[torch.arange(adv_logits.size(0)), labels] > 0.7
        #
        # outputs = outputs[logit_mask]
        # seq_outputs = sequence_mask_output[logit_mask]
        # new_label = labels[logit_mask]
        # #
        # #
        # rand_perm = torch.randperm(outputs.size(0))
        # rand_outputs = outputs[rand_perm, :]
        # rand_label = new_label[rand_perm]
        # pair_label = (new_label == rand_label).long()
        #
        # seq_outputs = self.map(seq_outputs)
        # rand_outputs = self.map(rand_outputs)

        pair_labels = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()

        # import  pdb
        # pdb.set_trace()

        contra_loss = self.contra_lc(
            sequence_mask_output_1.unsqueeze(1),
            sequence_mask_output_2.unsqueeze(0),
            pair_labels,
        )

        if torch.isnan(contra_loss):
            return 0

        return contra_loss

    def get_adv_loss(
        self,
        input_ids=None,
        attention_mask=None,
        mask_pos=None,
        labels=None,
        inputs_embeds=None,
    ):

        logits, sequence_mask_output = self.encode(
            input_ids, attention_mask, mask_pos, inputs_embeds
        )

        input_args = [input_ids, attention_mask, mask_pos, labels, None, 1]
        embed = self.forward(*input_args)
        noise = generate_noise(embed)

        for step in range(0, self.K):
            vat_args = [input_ids, attention_mask, mask_pos, labels, embed + noise, 2]
            adv_logits, _ = self.forward(*vat_args)
            adv_loss = stable_kl(adv_logits, logits.detach(), reduce=False)
            try:
                (delta_grad,) = torch.autograd.grad(
                    adv_loss, noise, only_inputs=True, retain_graph=False
                )
            except:
                import pdb

                pdb.set_trace()

            norm = delta_grad.norm()
            if torch.isnan(norm) or torch.isinf(norm):
                return 0
            eff_delta_grad = delta_grad * self.step_size
            delta_grad, eff_noise = norm_grad(delta_grad, eff_grad=eff_delta_grad)
            noise = noise + delta_grad * self.step_size
            # noise, eff_noise = self._norm_grad(delta_grad, eff_grad=eff_delta_grad, sentence_level=self.norm_level)
            noise = noise.detach()
            noise.requires_grad_()

        vat_args = [input_ids, attention_mask, mask_pos, labels, embed + noise, 2]

        adv_logits, sequence_mask_output = self.forward(*vat_args)
        # ori_args = model(*ori_args)
        # aug_args = [input_ids, token_type_ids, attention_mask, premise_mask, hyp_mask, task_id, 2, (embed + native_noise).detach()]

        adv_loss = self.adv_lc(adv_logits, logits)
        return adv_loss

    def embed_encode(self, input_ids):
        embedding_output = self.deberta.embeddings.word_embeddings(input_ids)
        return embedding_output

    def encode(
        self,
        input_ids=None,
        attention_mask=None,
        mask_pos=None,
        inputs_embeds=None,
        return_full_softmax=False,
    ):
        batch_size = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # Encode everything
        if inputs_embeds is None:
            outputs = self.deberta(input_ids, attention_mask=attention_mask)
        else:
            outputs = self.deberta(
                None, attention_mask=attention_mask, inputs_embeds=inputs_embeds
            )

        # Get <mask> token representation
        sequence_output = outputs[0]
        sequence_mask_output = sequence_output[
            torch.arange(sequence_output.size(0)), mask_pos
        ]

        # Logits over vocabulary tokens
        prediction_mask_scores = self.cls(sequence_mask_output)

        # sequence_mask_output = self.lm_head.dense(sequence_mask_output)

        # Exit early and only return mask logits.
        if return_full_softmax:
            return prediction_mask_scores

        # Return logits for each label
        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(
                prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1)
            )
        logits = torch.cat(logits, -1)

        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits)  # Log prob of right polarity

        return logits, sequence_mask_output

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mask_pos=None,
        labels=None,
        inputs_embeds=None,
        fwd_type=0,
        block_flag=None,
    ):
        if fwd_type == 2:
            assert inputs_embeds is not None
            return self.encode(
                input_ids=input_ids,
                attention_mask=attention_mask,
                mask_pos=mask_pos,
                inputs_embeds=inputs_embeds,
            )

        elif fwd_type == 1:
            return self.embed_encode(input_ids)

        logits, sequence_mask_output = self.encode(
            input_ids, attention_mask, mask_pos, inputs_embeds
        )

        loss = None

        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack(
                    [
                        1 - (labels.view(-1) - self.lb) / (self.ub - self.lb),
                        (labels.view(-1) - self.lb) / (self.ub - self.lb),
                    ],
                    -1,
                )
                loss = loss_fct(logits.view(-1, 2), labels)
            else:

                if labels.shape == logits.shape:
                    loss = F.kl_div(
                        F.log_softmax(logits, dim=-1, dtype=torch.float32),
                        labels,
                        reduction="batchmean",
                    )
                else:
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
                    # if self.model_args.hybrid == 1:
                    #     logits = logits[0]
                    #     cls_logits = logits[1]

                    #     cls_loss = loss_fct(cls_logits.view(-1, cls_logits.size(-1)), labels.view(-1))
                    #     loss = loss + cls_loss

        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (
                torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,
            )

        return ((loss,) + output) if loss is not None else output
