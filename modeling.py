
import torch
import torch.nn as nn
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import AutoModel, AutoConfig, PreTrainedModel, AutoModelForCausalLM, AutoModelForMaskedLM
from typing import Callable, List, Optional, Tuple, Union
from transformers.cache_utils import Cache


class ClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.vocab_size, config.hidden_size)
        classifier_dropout = 0.0
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features,mask_positions,  **kwargs):
        #print('mask_positions', mask_positions)
        x = features[:, mask_positions, :]  # Use [MASK] token
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class MLMSequenceClassification(PreTrainedModel):
    def __init__(self, config, transformer, mask_token_id =None, mask_last_pos = None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        if mask_token_id is not None:
            self.mask_token_id = mask_token_id
        elif mask_last_pos is not None:
            self.mask_last_pos = mask_last_pos
            self.mask_token_id=None
        else:
            self.mask_last_pos = -1
            self.mask_token_id=None

        self.transformer = transformer
        self.classifier = ClassificationHead(config)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        #print('input_ids', input_ids)

        if self.mask_token_id is not None:
            mask_positions = (input_ids == self.mask_token_id).nonzero(as_tuple=False)
            self.mask_positions = mask_positions[0][-1] - input_ids.shape[1]
            self.mask_token_id=None
            print('Got mask position: ', self.mask_positions)


        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            #head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs.logits
        logits = self.classifier(sequence_output, self.mask_positions)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif labels.dtype in [torch.long, torch.int]:
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze()) if self.num_labels == 1 else loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        config = AutoConfig.from_pretrained(model_name_or_path, **kwargs)
        transformer = AutoModelForMaskedLM.from_pretrained(model_name_or_path,trust_remote_code=True, config=config)
        return cls(config=config, transformer=transformer, mask_token_id = kwargs['mask_token_id'])



class CLMClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.vocab_size, config.hidden_size)
        classifier_dropout = 0.05
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features,  **kwargs):
        x = features[:, -1, :]  # Use [last] token
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class CLMSequenceClassification(PreTrainedModel):
    def __init__(self, config, transformer, mask_token_id =None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.transformer = transformer
        self.classifier = CLMClassificationHead(config)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        output_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
   
        )-> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        #print('input_ids', input_ids)


        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            
        )

        sequence_output = outputs.logits
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif labels.dtype in [torch.long, torch.int]:
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze()) if self.num_labels == 1 else loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        config = AutoConfig.from_pretrained(model_name_or_path, **kwargs)
        transformer = AutoModelForCausalLM.from_pretrained(model_name_or_path,trust_remote_code=True, config=config)
        return cls(config=config, transformer=transformer)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, config=None):
        # Load config if not provided
        if config is None:
            config = AutoConfig.from_pretrained(checkpoint_path)
        
        # Load transformer model separately
        transformer = AutoModel.from_pretrained(checkpoint_path, config=config)
        
        # Now correctly initialize your model with transformer
        model = cls(config=config, transformer=transformer)
    
        # Load the full model's state dict
        state_dict = torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"), map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
        if missing_keys:
            print(f"Warning: Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys: {unexpected_keys}")
    
        return model

