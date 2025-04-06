from comet import download_model, load_from_checkpoint
from comet.models.multitask.unified_metric import UnifiedMetric
from comet.models.utils import Prediction
from typing import Dict, Optional

import torch
import torch.nn as nn  # <-- Add this
class CustomXCOMET(UnifiedMetric):
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
            print("i am inside when len of the batch is 3")
            predictions = [self.forward(**input_seq) for input_seq in batch]
            print("predictions: ", predictions)
            avg_scores = torch.stack([pred.score for pred in predictions], dim=0).mean(dim=0)
            #print("avg scores", avg_scores)
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
                print("subword probs :", subword_probs)
                
                #### get word level probability

                tokenizer = self.encoder.tokenizer
                # # ====== Get original words from batch
                # 1. Extract tokens from the original batch
                original_subwords = self.encoder.tokenizer.convert_ids_to_tokens(
                        batch[0]["input_ids"][0],
                        skip_special_tokens=True
                )
                # 2. Get word IDs from the original tokenization (via mt_offsets)
                # mt_offsets is a list of (start, end) tuples for each subword
                # We'll group subwords that are part of the same original word
                original_words = []
                current_word = []
                current_end = 0
                for offset in batch[0]["mt_offsets"][0]:
                    start, end = offset
                    if start >= current_end:  # New word starts
                        if current_word:
                            original_words.append(current_word)
                        current_word = [original_subwords[len(original_words)]]
                        current_end = end
                    else:
                         # Continuation of current word (e.g., hyphenated or split subword)
                         current_word.append(original_subwords[len(original_words)])
                         current_end = end
                # Add the last word
                if current_word:
                    original_words.append(current_word)
                # Merge subwords into full words (e.g., ["Tory", "-", "Abgeordneten"] → "Tory-Abgeordneten")
                merged_words = [
                    self.encoder.tokenizer.convert_tokens_to_string(word).strip()
                    for word in original_words
                ]  
                # ====== Align probabilities with merged words ======
                # Group subword probabilities by original words
                word_probs = {}
                for idx, (start, end) in enumerate(batch[0]["mt_offsets"][0]):
                    word_idx = len(word_probs)  # Index of the current word
                    if idx >= len(subword_probs[0]):
                        break  # Handle padding
                    if word_idx not in word_probs:
                        word_probs[word_idx] = []
                    word_probs[word_idx].append(subword_probs[0][idx].cpu().numpy())
                # Aggregate probabilities (max for each class)
                token_level_probs = []
                for word_idx in sorted(word_probs.keys()):
                    probs = torch.stack([torch.tensor(p) for p in word_probs[word_idx]])
                    max_probs, _ = torch.max(probs, dim=0)
                    token_level_probs.append(max_probs.numpy())
                # Map merged words to probabilities
                token_predictions = [
                        {"token": word, "probabilities": probs.tolist()}
                        for word, probs in zip(merged_words, token_level_probs)
                ]
                print("Word-Level Probabilities:")
                for pred in token_predictions:
                    print(f"{pred['token']}: {pred['probabilities']}")


                ## create error span using decode function
                error_spans = self.decode(
                    subword_probs, batch[0]["input_ids"], batch[0]["mt_offsets"]
                )
                batch_prediction.metadata["error_spans"] = error_spans
        else:
            print("i am inside when len of the batch is not 3")
            model_output = self.forward(**batch[0])
            batch_prediction = Prediction(scores=model_output.score)
            if self.word_level:
                mt_mask = batch[0]["label_ids"] != -1
                mt_length = mt_mask.sum(dim=1)
                seq_len = mt_length.max()
                subword_probs = nn.functional.softmax(model_output.logits, dim=2)[:, :seq_len, :]
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

model = CustomXCOMET.load_from_checkpoint(path,strict = False)
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
