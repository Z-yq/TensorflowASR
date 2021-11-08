import tensorflow as tf
import tensorflow.keras.mixed_precision.experimental as mixed_precision
from punc_recover.models.punc_transformer import PuncTransformer
from punc_recover.trainer.base_trainers import BaseTrainer
from utils.text_featurizers import TextFeaturizer
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class PuncTrainer(BaseTrainer):
    """ Trainer for CTC Models """

    def __init__(self,

                 config: dict,
                 is_mixed_precision: bool = False,
                 strategy=None
                 ):
        super(PuncTrainer, self).__init__(config=config['running_config'],)

        self.is_mixed_precision = is_mixed_precision
        self.set_strategy(strategy)
        self.model_config=config['model_config']
        self.vocab_featurizer = TextFeaturizer(config['punc_vocab'])
        self.bd_featurizer = TextFeaturizer(config['punc_biaodian'])
        self.opt_config = config['optimizer_config']
    def creat_mask(self, seq):
        seq_pad = tf.cast(tf.equal(seq, 0), tf.float32)
        return seq_pad[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

    def set_train_metrics(self):
        lists=['bd_loss','feature_map_loss','bd_acc']

        self.train_metrics={}
        for item in lists:
            self.train_metrics.update({
            item: tf.keras.metrics.Mean("train_{}".format(item), dtype=tf.float32)
        })

    def set_eval_metrics(self):
        lists = ['bd_loss','feature_map_loss','bd_acc']

        self.eval_metrics={}
        for item in lists:
            self.eval_metrics.update({
            item: tf.keras.metrics.Mean("eval_{}".format(item), dtype=tf.float32),
        })

    @tf.function(experimental_relax_shapes=True)
    def _train_step(self, batch):
        inp, tar_bd,feature= batch


        mask=self.creat_mask(inp)
        with tf.GradientTape() as tape:
            bd_pred, out_feature = self.model([inp,mask],
                                                  training=True)

            bd_loss=self.classes_loss(tar_bd,bd_pred)
            feature_map_loss=self.bert_feature_loss(feature, out_feature)
            train_loss =feature_map_loss*10.+bd_loss

            if self.is_mixed_precision:
                scaled_train_loss = self.optimizer.get_scaled_loss(train_loss)
            train_loss = tf.nn.compute_average_loss(train_loss,
                                                    global_batch_size=self.global_batch_size)
        if self.is_mixed_precision:
            scaled_gradients = tape.gradient(scaled_train_loss, self.model.trainable_variables)
            gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = tape.gradient(train_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_metrics['feature_map_loss'].update_state(feature_map_loss)
        self.train_metrics['bd_loss'].update_state(bd_loss)
        self.train_metrics['bd_acc'].update_state(self.classes_acc(tar_bd,bd_pred))

    @tf.function(experimental_relax_shapes=True)
    def _eval_step(self, batch):
        inp, tar_bd, feature = batch


        mask=self.creat_mask(inp)
        pred_bd,out_feature = self.model([inp,mask],training=False)

        feature_map_loss = self.bert_feature_loss(feature, out_feature)
        bd_loss=self.classes_loss(tar_bd,pred_bd)

        self.eval_metrics['feature_map_loss'].update_state(feature_map_loss)
        self.eval_metrics['bd_loss'].update_state(bd_loss)
        self.eval_metrics['bd_acc'].update_state(self.classes_acc(tar_bd,pred_bd))


    def bert_feature_loss(self, real, pred):
        T1=tf.shape(real)[1]
        T2=tf.shape(pred)[1]
        T=tf.reduce_min([T1,T2])
        real=real[:,:T]
        pred=pred[:,:T]
        mask = tf.cast(tf.not_equal(real, -10.), 1.)
        loss = tf.square(real - pred)
        loss *= mask
        return tf.reduce_mean(tf.reduce_sum(loss,-1) / (tf.reduce_sum(mask,-1) + 1e-6),-1,True)

    def classes_loss(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss = tf.keras.losses.sparse_categorical_crossentropy(real, pred,True)
        mask_one= tf.math.logical_not(tf.math.equal(real, 1))
        mask = tf.cast(mask, dtype=loss.dtype)
        mask_one = tf.cast(mask_one, dtype=loss.dtype)
        mask_one*=mask
        loss_all =loss* mask
        final=tf.reduce_sum(loss_all,-1,True)/(tf.reduce_sum(mask,-1,True)+1e-6)
        loss_other = loss_all*mask_one
        final2 = tf.reduce_sum(loss_other, -1, True) / (tf.reduce_sum(mask_one, -1, True)+1e-6)
        return final+final2


    def classes_acc(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        accs = tf.keras.metrics.sparse_categorical_accuracy(real,pred)

        mask = tf.cast(mask, dtype=accs.dtype)
        accs *= mask
        final=tf.reduce_sum(accs,-1)/tf.reduce_sum(mask,-1)

        return tf.reduce_mean(final)

    def _eval_batches(self):
        """One epoch evaluation."""

        for metric in self.eval_metrics.keys():
            self.eval_metrics[metric].reset_states()

        for idx, batch in enumerate(self.eval_datasets):

            self.strategy.run(self._eval_step, args=(batch,))

            self.eval_progbar.update(1)
            self._print_eval_metrics(self.eval_progbar)
            if idx >= self.eval_steps_per_epoch:
                break
        self._write_to_tensorboard(self.eval_metrics, self.steps, stage="eval")
    def _train_batches(self):
        """Train model one epoch."""

        for idx,batch in enumerate(self.train_datasets):

            self.strategy.run(self._train_step,args=(batch,))
            self.steps+=1
            self.train_progbar.update(1)
            self._print_train_metrics(self.train_progbar)
            self._check_log_interval()
            self._check_save_interval()


            if idx>self.train_steps_per_epoch:
                break
    def compile(self,
                max_to_keep: int = 10):
        with self.strategy.scope():
            self.model=PuncTransformer(num_layers=self.model_config['num_layers'],
                                       d_model=self.model_config['d_model'],
                                       enc_embedding_dim=self.model_config['enc_embedding_dim'],
                                       num_heads=self.model_config['num_heads'],
                                       dff=self.model_config['dff'],
                                       input_vocab_size=self.vocab_featurizer.num_classes,
                                       bd_vocab_size=self.bd_featurizer.num_classes,
                                       pe_input=self.model_config['pe_input'] ,
                                       rate=self.model_config['rate'] )
            self.model._build()

            try:
                self.load_checkpoint()
            except:
                logging.info('trainer resume failed')
            self.model.summary(line_length=100)

            self.optimizer = tf.keras.optimizers.Adam(lr=self.opt_config['lr'], beta_1=self.opt_config['beta1'],
                                                      beta_2=self.opt_config['beta2'],
                                                      epsilon=self.opt_config['epsilon'])
            if self.is_mixed_precision:
                self.optimizer = mixed_precision.LossScaleOptimizer(self.optimizer, "dynamic")
        self.set_progbar()
        self.max_to_keep = max_to_keep

    def fit(self, epoch=None):
        if epoch is not None:
            self.epochs=epoch
            self.train_progbar.set_description_str(
                f"[Train] [Epoch {epoch}/{self.running_config['num_epochs']}]")


        self._train_batches()

        self._check_eval_interval()
            # self._eval_batches(eval_dataset)


