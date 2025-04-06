from comet import download_model, load_from_checkpoint
from comet.models.multitask.unified_metric import UnifiedMetric
from comet.models.utils import Prediction
from typing import Dict, Optional
import torch

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from torch import nn
from transformers.optimization import (Adafactor,
                                       get_constant_schedule_with_warmup)

from comet.models.base import CometModel
from comet.models.metrics import MCCMetric, RegressionMetrics
from comet.models.utils import LabelSet, Prediction, Target
from comet.modules import FeedForward

# Dummy replacement for model.predict
def my_predict(data, batch_size=8, gpus=1):
    def decode(
        self,
        subword_probs: torch.Tensor,
        input_ids: torch.Tensor,
        mt_offsets: torch.Tensor,
    ) -> List[Dict]:
        """Decode error spans from subwords.

        Args:
            subword_probs (torch.Tensor): probabilities of each label for each subword.
            input_ids (torch.Tensor): input ids from the model.
            mt_offsets (torch.Tensor): subword offsets.

        Return:
            List with of dictionaries with text, start, end, severity and a
            confidence score which is the average of the probs for that label.
        """
        decoded_output = []
        for i in range(len(mt_offsets)):
            seq_len = len(mt_offsets[i])
            error_spans, in_span, span = [], False, {}
            for token_id, probs, token_offset in zip(
                input_ids[i, :seq_len], subword_probs[i][:seq_len], mt_offsets[i]
            ):
                if self.decoding_threshold:
                    if torch.sum(probs[1:]) > self.decoding_threshold:
                        probability, label_value = torch.topk(probs[1:], 1)
                        label_value += 1  # offset from removing label 0
                    else:
                        # This is just to ensure same format but at this point
                        # we will only look at label 0 and its prob
                        probability, label_value = torch.topk(probs[0], 1)
                else:
                    probability, label_value = torch.topk(probs, 1)

                # Some torch versions topk returns a shape 1 tensor with only
                # a item inside
                label_value = (
                    label_value.item()
                    if label_value.dim() < 1
                    else label_value[0].item()
                )
                label = self.label_encoder.ids_to_label.get(label_value)
                # Label set:
                # O I-minor I-major
                # Begin of annotation span
                if label.startswith("I") and not in_span:
                    in_span = True
                    span["tokens"] = [
                        token_id,
                    ]
                    span["severity"] = label.split("-")[1]
                    span["offset"] = list(token_offset)
                    span["confidence"] = [
                        probability,
                    ]

                # Inside an annotation span
                elif label.startswith("I") and in_span:
                    span["tokens"].append(token_id)
                    span["confidence"].append(probability)
                    # Update offset end
                    span["offset"][1] = token_offset[1]

                # annotation span finished.
                elif label == "O" and in_span:
                    error_spans.append(span)
                    in_span, span = False, {}

            sentence_output = []
            for span in error_spans:
                sentence_output.append(
                    {
                        "text": self.encoder.tokenizer.decode(span["tokens"]),
                        "confidence": torch.concat(span["confidence"]).mean().item(),
                        "severity": span["severity"],
                        "start": span["offset"][0],
                        "end": span["offset"][1],
                    }
                )
            decoded_output.append(sentence_output)
        return decoded_output

    def predict_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
    ) -> Prediction:
        """PyTorch Lightning predict_step

        Args:
            batch (Dict[str, torch.Tensor]): The output of your prepare_sample function
            batch_idx (Optional[int], optional): Integer displaying which batch this is
                Defaults to None.
            dataloader_idx (Optional[int], optional): Integer displaying which
                dataloader this is. Defaults to None.

        Returns:
            Prediction: Model Prediction
        """
        if len(batch) == 3:
            predictions = [self.forward(**input_seq) for input_seq in batch]
            # Final score is the average of the 3 scores!
            avg_scores = torch.stack([pred.score for pred in predictions], dim=0).mean(
                dim=0
            )
            batch_prediction = Prediction(
                scores=avg_scores,
                metadata=Prediction(
                    src_scores=predictions[0].score,
                    ref_scores=predictions[1].score,
                    unified_scores=predictions[2].score,
                ),
            )
            if self.word_level:
                mt_mask = batch[0]["label_ids"] != -1
                mt_length = mt_mask.sum(dim=1)
                seq_len = mt_length.max()
                subword_probs = [
                    nn.functional.softmax(o.logits, dim=2)[:, :seq_len, :] * w
                    for w, o in zip(self.input_weights_spans, predictions)
                ]
                subword_probs = torch.sum(torch.stack(subword_probs), dim=0)
                error_spans = self.decode(
                    subword_probs, batch[0]["input_ids"], batch[0]["mt_offsets"]
                )
                batch_prediction.metadata["error_spans"] = error_spans

        else:
            model_output = self.forward(**batch[0])
            batch_prediction = Prediction(scores=model_output.score)
            if self.word_level:
                mt_mask = batch[0]["label_ids"] != -1
                mt_length = mt_mask.sum(dim=1)
                seq_len = mt_length.max()
                subword_probs = nn.functional.softmax(model_output.logits, dim=2)[
                    :, :seq_len, :
                ]
                error_spans = self.decode(
                    subword_probs, batch[0]["input_ids"], batch[0]["mt_offsets"]
                )
                batch_prediction = Prediction(
                    scores=model_output.score,
                    metadata=Prediction(error_spans=error_spans),
                )
        return batch_prediction
# Load checkpoint into your custom class
path = "/storage/brno2/home/rahmang/xcomet/downloadedxcomet/models--Unbabel--XCOMET-XL/snapshots/50d428488e021205a775d5fab7aacd9502b58e64/checkpoints/model.ckpt"

model = load_from_checkpoint(path)
model.predict = my_predict
data = [
    {
        "src": "Boris Johnson teeters on edge of favour with Tory MPs",
        "mt": "Boris Johnson ist bei Tory-Abgeordneten völlig in der Gunst",
        "ref": "Boris Johnsons Beliebtheit bei Tory-MPs steht auf der Kippe"
    }
]
model_output = model.predict(data, batch_size=8, gpus=1)
# Segment-level scores
print (model_output.scores)

# System-level score
print (model_output.system_score)

# Score explanation (error spans)
print (model_output.metadata.error_spans)
