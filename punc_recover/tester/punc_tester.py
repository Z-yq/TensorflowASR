import logging
import os
import tensorflow as tf
from punc_recover.models.punc_transformer import PuncTransformer
from punc_recover.tester.base_tester import BaseTester
from utils.text_featurizers import TextFeaturizer


class PuncTester(BaseTester):
    """ Trainer for CTC Models """

    def __init__(self,
                 config,

                 ):
        super(PuncTester, self).__init__(config['running_config'])

        self.model_config = config['model_config']
        self.vocab_featurizer = TextFeaturizer(config['punc_vocab'])
        self.bd_featurizer = TextFeaturizer(config['punc_biaodian'])
        self.opt_config = config['optimizer_config']
        self.eval_metrics = {
            "acc": tf.keras.metrics.Mean(),

        }

    def _eval_step(self, batch):
        x, labels = batch


        mask = self.creat_mask(x)
        pred_bd = self.model.inference(x, mask)
        acc=self.classes_acc(labels,pred_bd)
        self.eval_metrics["acc"].update_state(acc)

    def creat_mask(self, seq):
        seq_pad = tf.cast(tf.equal(seq, 0), tf.float32)
        return seq_pad[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

    def classes_acc(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        accs = tf.keras.metrics.sparse_categorical_accuracy(real,pred)

        mask = tf.cast(mask, dtype=accs.dtype)
        accs *= mask
        final=tf.reduce_sum(accs,-1)/tf.reduce_sum(mask,-1)

        return tf.reduce_mean(final)
    def compile(self, ):
        self.model = PuncTransformer(num_layers=self.model_config['num_layers'],
                                     d_model=self.model_config['d_model'],
                                     enc_embedding_dim=self.model_config['enc_embedding_dim'],
                                     num_heads=self.model_config['num_heads'],
                                     dff=self.model_config['dff'],
                                     input_vocab_size=self.vocab_featurizer.num_classes,
                                     bd_vocab_size=self.bd_featurizer.num_classes,
                                     pe_input=self.model_config['pe_input'],
                                     rate=self.model_config['rate'])
        self.model._build()

        self.load_checkpoint()

        logging.info('trainer resume failed')
        self.model.summary(line_length=100)


    def run(self, ):
        self._eval_batches()

    def load_checkpoint(self, ):
        """Load checkpoint."""

        self.checkpoint_dir = os.path.join(self.running_config["outdir"], "checkpoints")
        files = os.listdir(self.checkpoint_dir)
        files.sort(key=lambda x: int(x.split('_')[-1].replace('.h5', '')))
        self.model.load_weights(os.path.join(self.checkpoint_dir, files[-1]))
        self.steps = int(files[-1].split('_')[-1].replace('.h5', ''))
