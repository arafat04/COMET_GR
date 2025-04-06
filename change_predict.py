from comet import download_model, load_from_checkpoint

from comet.models.multitask.unified_metric import UnifiedMetric  # Actual 

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
            # Final score is the average of the 3 scores!
            avg_scores = torch.stack([pred.score for pred in predictions], dim=0).mean(
                dim=0
            )
            
            print("avg scores", avg_scores)
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
            print("i am inside when len of the batch is not 3")
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

model = CustomXCOMET.load_from_checkpoint(path)
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
