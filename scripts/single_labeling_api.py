import warnings
import os

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
###################################################################
import numpy as np
import tensorflow as tf
import random as rn

np.random.seed(42)
rn.seed(12345)
from keras import backend as K

tf.set_random_seed(1234)
####################################################################

from twilbert.models.bert import BertModel as TWilBertModel
from twilbert.preprocessing.tokenizer import FullTokenizer
from twilbert.optimization.lr_annealing import Noam
from twilbert.utils.utils import Utils as ut
from twilbert.utils.generator import SingleFinetuningGenerator
from twilbert.models.finetuning_models import finetune_ffn
from twilbert.utils.finetuning_monitor import FinetuningMonitor
from tqdm import trange
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
import collections
import json
import sys


class TWilBertLabelClass(object):
    """Apply a TWilBert finetuned model over new samples"""

    def predict(self, text_list):
        """Perform

        :param text_list: a list of texts

        """

        rev_categories = self.parameters["rev_categories"]
        ids_ts = range(len(text_list))
        x_ts = text_list

        # pick one entry of the list of classes (it is not relevant as we are going to predict the value anyway)
        default_class = list(rev_categories.values())[0]

        y_ts = [default_class for i in range(len(text_list))]

        tokenizer = self.parameters["tokenizer"]
        n_classes = self.parameters["n_classes"]
        bucket_min = self.parameters["bucket_min"]
        bucket_max = self.parameters["bucket_max"]
        bucket_steps = self.parameters["bucket_steps"]
        preprocessing = self.parameters["preprocessing"]
        multi_label = self.parameters["multi_label"]

        gen_ts = SingleFinetuningGenerator(
            tokenizer,
            n_classes,
            bucket_min,
            bucket_max,
            bucket_steps,
            preprocessing,
            multi_label,
        )

        ts_gen = gen_ts.generator(ids_ts, x_ts, y_ts)

        pred_batch_size = self.parameters["pred_batch_size"]
        n_buckets = self.parameters["n_buckets"]

        test_preds = []
        for b in range(n_buckets):
            (bx, by) = next(ts_gen)
            if len(bx[0]) == 0:
                continue

            preds = self.finetune_model.predict(x=bx, batch_size=pred_batch_size)

            preds = preds.tolist()
            test_preds += preds

        label_list = []

        for _pred in test_preds:
            id_class = np.argmax(_pred)
            label = rev_categories[str(id_class)]
            label_list.append(label)

        return label_list, test_preds

    def __init__(self, config_file):
        """Initialization of the model

        :param config_file:

        """

        with open(config_file, "r") as json_file:
            config = json.load(json_file)

        # Task CSV Headers, Categories, Unbalancing and Number of Classes #

        categories = config["task"]["categories"]
        rev_categories = config["task"]["rev_categories"]
        n_classes = len(categories)
        multi_label = config["task"]["multi_label"]

        #######################

        # Representation #

        vocab_file = config["dataset"]["vocab_file"]
        tokenizer = FullTokenizer(vocab_file)
        vocab_size = len(tokenizer.vocab)
        max_len = config["representation"]["max_len"]
        bucket_min = config["representation"]["bucket_min"]
        bucket_max = config["representation"]["bucket_max"]
        bucket_steps = config["representation"]["bucket_steps"]
        preprocessing = config["representation"]["preprocessing"]

        ##################

        # Finetuning model parameters #
        pred_batch_size = config["finetuning"]["pred_batch_size"]
        trainable_layers = config["finetuning"]["trainable_layers"]
        collapse_mode = config["finetuning"]["collapse_mode"]
        finetune_dropout = config["finetune_model"]["dropout"]
        loss = config["finetuning"]["loss"]
        pretrained_model_weights = config["finetuning"]["path_load_weights"]
        finetune_model_weights = config["finetuning"]["path_load_finetuned_weights"]
        lr = config["finetuning"]["lr"]
        optimizer = config["finetuning"]["optimizer"]
        accum_iters = config["finetuning"]["accum_iters"]

        ##############################

        # Pretrained TWilBert parameters #

        factorize_embeddings = config["model"]["factorize_embeddings"]
        cross_sharing = config["model"]["cross_sharing"]
        embedding_size = config["model"]["embedding_size"]
        hidden_size = config["model"]["hidden_size"]
        n_encoders = config["model"]["n_encoders"]
        n_heads = config["model"]["n_heads"]
        attention_size = config["model"]["attention_size"]
        attention_size = (
            hidden_size // n_heads if attention_size is None else attention_size
        )
        input_dropout = config["model"]["input_dropout"]
        output_dropout = config["model"]["output_dropout"]

        pkm = config["model"]["pkm"]
        pkm_params = config["model"]["pkm_params"]

        rop_n_hidden = config["model"]["rop"]["n_hidden"]
        rop_hidden_size = config["model"]["rop"]["hidden_size"]

        output_encoder_size = [hidden_size for i in range(n_encoders)]
        attention_size = [attention_size for i in range(n_encoders)]
        n_heads = [n_heads for i in range(n_encoders)]

        ##################################

        # Load TWilBert model #

        twilbert_model = TWilBertModel(
            max_len,
            vocab_size,
            embedding_size,
            output_encoder_size,
            attention_size,
            n_heads,
            cross_sharing,
            factorize_embeddings,
            input_dropout,
            output_dropout,
            rop_n_hidden,
            rop_hidden_size,
            optimizer,
            accum_iters,
            pkm,
            pkm_params,
            input_length=None,
        )

        twilbert_model.build()

        model = twilbert_model.model
        pretrained_model = twilbert_model.pretrained_model
        twilbert_model.compile(model)
        model.load_weights(pretrained_model_weights)

        n_buckets = int((bucket_max - bucket_min) / bucket_steps)

        # Load finetune model #
        # FIXME: to self
        finetune_model = finetune_ffn(
            pretrained_model,
            n_classes,
            trainable_layers,
            collapse_mode,
            finetune_dropout=finetune_dropout,
            loss=loss,
            lr=lr,
            multi_label=multi_label,
            optimizer=optimizer,
            accum_iters=accum_iters,
        )

        finetune_model.load_weights(finetune_model_weights)
        print(finetune_model.summary())

        # Store parameters to use them during inference
        self.parameters = {
            "tokenizer": tokenizer,
            "n_classes": n_classes,
            "bucket_min": bucket_min,
            "bucket_max": bucket_max,
            "bucket_steps": bucket_steps,
            "preprocessing": preprocessing,
            "multi_label": multi_label,
            "n_buckets": n_buckets,
            "rev_categories": rev_categories,
            "pred_batch_size": pred_batch_size,
        }

        self.finetune_model = finetune_model


if __name__ == "__main__":

    processor = TWilBertLabelClass(
        "configs/microservs/config_labelling_single_hateeval19_large.json"
    )
    print(processor.predict(["Ay wey cállate pinche puta"]))
