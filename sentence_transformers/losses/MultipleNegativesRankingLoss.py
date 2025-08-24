from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor, nn

from sentence_transformers import util
from sentence_transformers.SentenceTransformer import SentenceTransformer

import copy

def ed_calc(x, attention_mask):
    """
    Calculate the energy for all queries in parallel, accounting for padding.

    Args:
	x (torch.Tensor): Query embeddings of shape [num_queries, max_sequence_length, query_dim].
        attention_mask (torch.Tensor): Mask of shape [num_queries, max_sequence_length],
                                        where 1 indicates valid tokens and 0 indicates padding.

    Returns:
	torch.Tensor: Energy values for each query, shape [num_queries].
    """
    # Shape: [num_queries, max_sequence_length, query_dim]
    num_queries, max_sequence_length, query_dim = x.shape

    # Create pairwise differences: [num_queries, max_sequence_length, max_sequence_length, query_dim]
    x_expanded_1 = x.unsqueeze(2)  # Expand along the second dimension
    x_expanded_2 = x.unsqueeze(1)  # Expand along the third dimension
    pairwise_diff = x_expanded_1 - x_expanded_2

    # Compute pairwise distances: [num_queries, max_sequence_length, max_sequence_length]
    pairwise_distances = torch.norm(pairwise_diff, dim=3)

    # Apply the attention mask to exclude padded tokens
    attention_mask_expanded = attention_mask.unsqueeze(2) & attention_mask.unsqueeze(1)  # [num_queries, max_sequence_length, max_sequence_length]
    pairwise_distances = (pairwise_distances * attention_mask_expanded).clone()  # Mask padded positions

    # Count valid token pairs for normalization
    valid_pairs = attention_mask_expanded.sum(dim=(1, 2)).clamp(min=1)  # Shape: [num_queries]

    # Sum of distances and normalize
    ed_sums = pairwise_distances.sum(dim=(1, 2))  # Sum across all pairs
    energy = ed_sums / valid_pairs  # Normalize by the number of valid pairs
    #print("ed_calc_new result:", energy)
    return energy  # Shape: [num_queries]

def energy_distance_old(x, y, attention_mask):
    # Shape of x: [num_queries, max_sequence_length, query_dim]
    # Shape of y: [num_docs, doc_dim]
    #print("ED calculation tensors")
    #print(x.device)  # Check device
    #print(y.device)  # Check device

    num_queries, max_sequence_length, query_dim = x.shape
    num_docs, doc_dim = y.shape

    # Check for dimensionality compatibility
    assert query_dim == doc_dim, "Query and document dimensions must match!"

    # Pre-calculate energy for all queries (batch of 2D query tensors)
    ed_queries = ed_calc(x, attention_mask)

    # Expand query tensor for broadcasting:
    # x_expanded: [num_queries, num_docs, max_seq_length, query_dim]
    x_expanded = x.unsqueeze(1).expand(-1, num_docs, -1, -1)

    # Expand document tensor for broadcasting:
    # y_expanded: [num_queries, num_docs, query_dim] -> unsqueeze for broadcasting
    y_expanded = y.unsqueeze(0).expand(num_queries, -1, -1)

    # Now, x_expanded has shape [num_queries, num_docs, max_seq_length, query_dim]
    # y_expanded has shape [num_queries, num_docs, query_dim]

    # Calculate energy distances for all query-document pairs in parallel
    pairwise_diff = x_expanded - y_expanded.unsqueeze(2)  # Shape: [num_queries, num_docs, max_seq_length, query_dim]
    pairwise_distances = torch.norm(pairwise_diff, dim=3)

    #squared_distances = torch.sum(pairwise_diff ** 2, dim=3)  # Shape: [num_queries, num_docs, max_seq_length]

    # Apply attention mask to zero out padded embeddings
    # Expand attention_mask for broadcasting: [num_queries, max_sequence_length] -> [num_queries, 1, max_sequence_length]
    attention_mask_expanded = attention_mask.unsqueeze(1).expand(-1, num_docs, -1)
    pairwise_distances = (pairwise_distances * attention_mask_expanded).clone()

    # Compute the sum of sqrt of squared distances for each query-document pair
    #ed_sums = torch.sum(torch.sqrt(squared_distances), dim=2)  # Shape: [num_queries, num_docs]

    # Sum distances and normalize by valid token count for each query-document pair
    # valid_token_counts: [num_queries, 1, max_sequence_length] -> [num_queries, num_docs]
    valid_token_counts = attention_mask_expanded.sum(dim=2).clamp(min=1)  # Avoid division by zero
    ed_sums = torch.sum(pairwise_distances, dim=2) / valid_token_counts

    # Final energy distance calculation (using pre-calculated query energies)
    energy_distances = 2 * ed_sums - ed_queries.unsqueeze(1)

    return energy_distances

def energy_distance(x, y, attention_mask):
    """
    Compute energy distance between multivector queries and single vector documents using torch.einsum.

    Args:
        x (torch.Tensor): Query embeddings of shape [num_queries, max_sequence_length, query_dim].
        y (torch.Tensor): Document embeddings of shape [num_docs, doc_dim].
        attention_mask (torch.Tensor): Attention mask of shape [num_queries, max_sequence_length].

    Returns:
        torch.Tensor: Energy distances of shape [num_queries, num_docs].
    """
    # Shapes
    num_queries, max_sequence_length, query_dim = x.shape
    num_docs, doc_dim = y.shape

    # Ensure dimensions are compatible
    assert query_dim == doc_dim, "Query and document dimensions must match!"

    # Step 1: Compute squared norms of the document embeddings (efficient norm calculation)
    # y: [num_docs, query_dim], norm_y: [num_docs]
    norm_y = torch.einsum("nd,nd->n", y, y)  # Shape: [num_docs]

    # Step 2: Compute pairwise squared distances between query tokens and document embeddings
    # x: [num_queries, max_sequence_length, query_dim], y: [num_docs, query_dim]
    # Output shape: [num_queries, max_sequence_length, num_docs]
    dot_product = torch.einsum("qld,nd->qln", x, y)  # Shape: [num_queries, max_sequence_length, num_docs]

    # Squared distance calculation
    # norm_x_tokens: [num_queries, max_sequence_length]
    norm_x_tokens = torch.einsum("qld,qld->ql", x, x)  # Shape: [num_queries, max_sequence_length]
    
    # Applying the squared distance formula: ||x_i - y_j||^2 = ||x_i||^2 + ||y_j||^2 - 2 * <x_i, y_j>
    squared_distances = (
        norm_x_tokens.unsqueeze(2) + norm_y.unsqueeze(0).unsqueeze(1) - 2 * dot_product
    )  # Shape: [num_queries, max_sequence_length, num_docs]

    # Ensure distances are non-negative due to numerical instability
    squared_distances = squared_distances.clamp(min=0)

    # Step 3: Compute Euclidean distances (L2 norm)
    distances = torch.sqrt(squared_distances)  # Shape: [num_queries, max_sequence_length, num_docs]

    # Step 4: Apply attention mask to the distances
    attention_mask_expanded = attention_mask.unsqueeze(2)  # Shape: [num_queries, max_sequence_length, 1]
    masked_distances = distances * attention_mask_expanded  # Shape: [num_queries, max_sequence_length, num_docs]

    # Step 5: Aggregate distances across sequence length and normalize by valid token count
    valid_token_counts = attention_mask.sum(dim=1).clamp(min=1).unsqueeze(1)  # Shape: [num_queries, 1]
    ed_sums = masked_distances.sum(dim=1) / valid_token_counts  # Shape: [num_queries, num_docs]

    # Step 6: Compute energy for queries and combine with pairwise distances
    ed_queries = ed_calc(x, attention_mask)  # Precomputed energy for each query, shape: [num_queries]
    energy_distances = (2 * ed_sums - ed_queries.unsqueeze(1)) * -1  # Shape: [num_queries, num_docs]

    return energy_distances

class MultipleNegativesRankingLoss(nn.Module):
    def __init__(self, model: SentenceTransformer, scale: float = 20.0, similarity_fct=energy_distance) -> None:
        """
        This loss expects as input a batch consisting of sentence pairs ``(a_1, p_1), (a_2, p_2)..., (a_n, p_n)``
        where we assume that ``(a_i, p_i)`` are a positive pair and ``(a_i, p_j)`` for ``i != j`` a negative pair.

        For each ``a_i``, it uses all other ``p_j`` as negative samples, i.e., for ``a_i``, we have 1 positive example
        (``p_i``) and ``n-1`` negative examples (``p_j``). It then minimizes the negative log-likehood for softmax
        normalized scores.

        This loss function works great to train embeddings for retrieval setups where you have positive pairs
        (e.g. (query, relevant_doc)) as it will sample in each batch ``n-1`` negative docs randomly.

        The performance usually increases with increasing batch sizes.

        You can also provide one or multiple hard negatives per anchor-positive pair by structuring the data like this:
        ``(a_1, p_1, n_1), (a_2, p_2, n_2)``. Then, ``n_1`` is a hard negative for ``(a_1, p_1)``. The loss will use for
        the pair ``(a_i, p_i)`` all ``p_j`` for ``j != i`` and all ``n_j`` as negatives.

        Args:
            model: SentenceTransformer model
            scale: Output of similarity function is multiplied by scale
                value
            similarity_fct: similarity function between sentence
                embeddings. By default, cos_sim. Can also be set to dot
                product (and then set scale to 1)

        References:
            - Efficient Natural Language Response Suggestion for Smart Reply, Section 4.4: https://arxiv.org/pdf/1705.00652.pdf
            - `Training Examples > Natural Language Inference <../../examples/training/nli/README.html>`_
            - `Training Examples > Paraphrase Data <../../examples/training/paraphrases/README.html>`_
            - `Training Examples > Quora Duplicate Questions <../../examples/training/quora_duplicate_questions/README.html>`_
            - `Training Examples > MS MARCO <../../examples/training/ms_marco/README.html>`_
            - `Unsupervised Learning > SimCSE <../../examples/unsupervised_learning/SimCSE/README.html>`_
            - `Unsupervised Learning > GenQ <../../examples/unsupervised_learning/query_generation/README.html>`_

        Requirements:
            1. (anchor, positive) pairs or (anchor, positive, negative) triplets

        Inputs:
            +-------------------------------------------------+--------+
            | Texts                                           | Labels |
            +=================================================+========+
            | (anchor, positive) pairs                        | none   |
            +-------------------------------------------------+--------+
            | (anchor, positive, negative) triplets           | none   |
            +-------------------------------------------------+--------+
            | (anchor, positive, negative_1, ..., negative_n) | none   |
            +-------------------------------------------------+--------+

        Recommendations:
            - Use ``BatchSamplers.NO_DUPLICATES`` (:class:`docs <sentence_transformers.training_args.BatchSamplers>`) to
              ensure that no in-batch negatives are duplicates of the anchor or positive samples.

        Relations:
            - :class:`CachedMultipleNegativesRankingLoss` is equivalent to this loss, but it uses caching that allows for
              much higher batch sizes (and thus better performance) without extra memory usage. However, it is slightly
              slower.
            - :class:`MultipleNegativesSymmetricRankingLoss` is equivalent to this loss, but with an additional loss term.
            - :class:`GISTEmbedLoss` is equivalent to this loss, but uses a guide model to guide the in-batch negative
              sample selection. `GISTEmbedLoss` yields a stronger training signal at the cost of some training overhead.

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                model = SentenceTransformer("microsoft/mpnet-base")
                train_dataset = Dataset.from_dict({
                    "anchor": ["It's nice weather outside today.", "He drove to work."],
                    "positive": ["It's so sunny.", "He took the car to the office."],
                })
                loss = losses.MultipleNegativesRankingLoss(model)

                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__()
        self.model = model
        self.scale = scale
        print("scale: ", scale)
        self.similarity_fct = similarity_fct
        print("similarity fct: ", similarity_fct)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        #torch.autograd.set_detect_anomaly(True)  # This will catch NaNs or infinite values during backward
        #for idx, feature in enumerate(sentence_features):
        #    input_ids = feature["input_ids"]
        #    if input_ids.size(1) > self.model.max_seq_length:
        #        print(f"[ERROR] Input {idx} length={input_ids.size(1)} exceeds max_seq_len={self.model.max_seq_length}")
        # Compute the embeddings and distribute them to anchor and candidates (positive and optionally negatives)
        #embeddings = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        #anchors = embeddings[0]  # (batch_size, embedding_dim)
        #anchors = self.model(sentence_features[0])["token_embeddings"]
        #print("Anchor dimensions:", anchors.size())
        #print("Anchor requires_grad:", anchors.requires_grad)
        #print("anchors NaNs:", torch.isnan(anchors).any().item())
        #print("anchors Infs:", torch.isinf(anchors).any().item())
        #print("sentence_features[0] keys:", sentence_features[0].keys())
        #print("shape of input_ids:", sentence_features[0]["input_ids"].shape)
        #print("model output keys:", self.model(sentence_features[0]).keys())
        #print("input_ids min/max:", sentence_features[0]["input_ids"].min().item(), sentence_features[0]["input_ids"].max().item())
        #print("input_ids dtype:", sentence_features[0]["input_ids"].dtype)
        # Run the model only once per input and collect sentence embeddings
        model_outputs = [self.model(f) for f in sentence_features]
        embeddings = [output["sentence_embedding"] for output in model_outputs]  # List[Tensor]
        #for i, feature in enumerate(sentence_features):
        #    try:
        #        max_len = feature["input_ids"].size(1)
        #        max_token = feature["input_ids"].max().item()
        #        print(f"[DEBUG] Sample {i} - seq len: {max_len}, max token id: {max_token}")
        #    except Exception as e:
        #        print(f"[ERROR] Failed on input {i}: {e}")


        # Extract token_embeddings and attention_mask from the first output
        first_output = model_outputs[0]
        anchors = first_output["token_embeddings"]
        attention_mask = first_output["attention_mask"]

        candidates = torch.cat(embeddings[1:])  # (batch_size * (1 + num_negatives), embedding_dim)
        scores = self.similarity_fct(anchors, candidates, attention_mask) * self.scale * -1
        #try:
            #with torch.autograd.set_detect_anomaly(True):
        #    scores = self.similarity_fct(anchors, candidates, attention_mask) * self.scale * -1
        #except Exception as e:
        #    print("[ERROR in similarity_fct]")
        #    raise e 
       # (batch_size, batch_size * (1 + num_negatives))
        #print("Score tensor dimensions:", scores.size())
        #print("Scores requires_grad:", scores.requires_grad)
        # anchor[i] should be most similar to candidates[i], as that is the paired positive,
        # so the label for anchor[i] is i
        range_labels = torch.arange(0, scores.size(0), device=scores.device)

        loss = self.cross_entropy_loss(scores, range_labels)
        #print("Loss:", loss.item())
        #print("Loss requires_grad:", loss.requires_grad)
        #print("=== [END DEBUG] ===\n")
        return loss


    def get_config_dict(self) -> dict[str, Any]:
        return {"scale": self.scale, "similarity_fct": self.similarity_fct.__name__}

    @property
    def citation(self) -> str:
        return """
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""
