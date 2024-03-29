from abc import ABC

import torch


class Loss(ABC):
    def compute(self, *args, **kwargs):
        pass


class SpERTLoss(Loss):
    def __init__(self, rel_criterion, entity_criterion, model, optimizer, scheduler, max_grad_norm):
        self._rel_criterion = rel_criterion
        self._entity_criterion = entity_criterion
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._max_grad_norm = max_grad_norm

    def compute(self, entity_s_logits, entity_o_logits, rel_logits, entity_types, rel_types, entity_sample_masks, rel_sample_masks):
        # entity loss
        entity_s_logits = entity_s_logits.view(-1, entity_s_logits.shape[-1])  
        
        entity_o_logits = entity_o_logits.view(-1, entity_o_logits.shape[-1])


        
        entity_s_types = (entity_types == 1).view(-1).long()  
        entity_o_types = (entity_types == 2).view(-1).long()  
        

        entity_sample_masks = entity_sample_masks.view(-1).float()  

        entity_s_loss = self._entity_criterion(entity_s_logits, entity_s_types)  
        entity_s_loss = (entity_s_loss * entity_sample_masks).sum() / entity_sample_masks.sum()
        
        entity_o_loss = self._entity_criterion(entity_o_logits, entity_o_types)  
        entity_o_loss = (entity_o_loss * entity_sample_masks).sum() / entity_sample_masks.sum()
        

        # relation loss
        rel_sample_masks = rel_sample_masks.view(-1).float()  
        
        rel_count = rel_sample_masks.sum()  

        if rel_count.item() != 0:
            rel_logits = rel_logits.view(-1, rel_logits.shape[-1])   
            
            rel_types = rel_types.view(-1, rel_types.shape[-1])  
            
            rel_loss = self._rel_criterion(rel_logits, rel_types)
            rel_loss = rel_loss.sum(-1) / rel_loss.shape[-1]
            rel_loss = (rel_loss * rel_sample_masks).sum() / rel_count

            # joint loss
            train_loss = entity_s_loss + rel_loss + entity_o_loss
        else:
            # corner case: no positive/negative relation samples
            train_loss = entity_s_loss + entity_o_loss

        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
        self._optimizer.step()
        self._scheduler.step()
        self._model.zero_grad()
        return train_loss.item()
