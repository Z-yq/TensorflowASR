import logging
import os
import tensorflow as tf
import tensorflow.keras.mixed_precision.experimental as mixed_precision

from asr.models.conformer_blocks import ConformerEncoder, StreamingConformerEncoder, CTCDecoder, Translator
from asr.trainer.base_runners import BaseTrainer
from utils.text_featurizers import TextFeaturizer


class CTCTrainer(BaseTrainer):
    """ Trainer for CTC Models """

    def __init__(self,

                 config: dict,
                 is_mixed_precision: bool = False,
                 strategy=None
                 ):
        super(CTCTrainer, self).__init__(config=config['running_config'], )
        self.config = config
        self.speech_config = config['speech_config']
        self.model_config = config['model_config']
        self.opt_config=config['optimizer_config']
        self.phone_featurizer = TextFeaturizer(config['inp_config'])
        self.text_featurizer = TextFeaturizer(config['tar_config'])
        self.is_mixed_precision = is_mixed_precision
        self.set_strategy(strategy)

    def set_train_metrics(self):
        self.train_metrics = {
            "train_loss": tf.keras.metrics.Mean("train_loss", dtype=tf.float32),
            "ctc_loss": tf.keras.metrics.Mean("train_ctc_loss", dtype=tf.float32),
            "translate_loss": tf.keras.metrics.Mean("translate_loss", dtype=tf.float32),
            "ctc_acc": tf.keras.metrics.Mean("train_ctc_acc", dtype=tf.float32),
            "translate_acc": tf.keras.metrics.Mean("translate_acc", dtype=tf.float32),

        }

    def set_eval_metrics(self):
        self.eval_metrics = {
            "ctc_loss": tf.keras.metrics.Mean("eval_ctc_loss", dtype=tf.float32),
            "ctc_acc": tf.keras.metrics.Mean("eval_ctc_acc", dtype=tf.float32),
            "translate_acc": tf.keras.metrics.Mean("translate_acc", dtype=tf.float32),
            "translate_loss": tf.keras.metrics.Mean("translate_loss", dtype=tf.float32),
        }

    def ctc_acc(self, labels, y_pred):
        T1 = tf.shape(y_pred)[1]
        T2 = tf.shape(labels)[1]
        T = tf.reduce_min([T1, T2])
        y_pred = y_pred[:, :T]
        labels = labels[:, :T]

        mask = tf.cast(tf.not_equal(labels, self.text_featurizer.pad), 1.)
        y_pred = tf.cast(y_pred, tf.float32)
        labels = tf.cast(labels, tf.float32)

        value = tf.cast(labels == y_pred, tf.float32)

        accs = tf.reduce_sum(value*mask, -1) / (tf.reduce_sum(mask, -1) + 1e-6)
        return accs
    def translate_acc(self,label,y_pred,length):
        need=tf.cast(tf.where(label==0,0,1),tf.float32)
        label=tf.cast(label,tf.float32)
        y_pred=tf.cast(tf.argmax(y_pred,-1),tf.float32)
        acc=tf.cast(label==y_pred[:,:length],tf.float32)
        return tf.reduce_sum(acc*need)/(tf.reduce_sum(need)+1e-6)
    def mask_loss(self,label,y_pred):

        need=tf.cast(tf.where(label==0,0,1),tf.float32)
        zero=tf.cast(tf.where(label==0,1,0),tf.float32)
        loss=tf.keras.losses.sparse_categorical_crossentropy(label,y_pred,True)
        need_loss=tf.reduce_sum(loss*need)/(tf.reduce_sum(need)+1e-6)
        zero_loss=tf.reduce_sum(loss*zero)/(tf.reduce_sum(zero)+1e-6)
        return tf.reduce_mean(loss,-1)+need_loss+zero_loss


    @tf.function(experimental_relax_shapes=True)
    def _train_step(self, batch):
        features, input_length, phone_labels, phone_label_length, tar_label = batch

        max_length=tf.shape(tar_label)[1]
        with tf.GradientTape() as tape:

            enc_output = self.encoder(features, training=True)
            ctc_output = self.ctc_model(enc_output, training=True)
            #print(enc_output.shape,ctc_output.shape,phone_labels.shape,input_length.shape,phone_label_length.shape)

            ctc_output = tf.nn.softmax(ctc_output, -1)
            ctc_loss = tf.keras.backend.ctc_batch_cost(tf.cast(phone_labels, tf.int32),
                                            tf.cast(ctc_output, tf.float32),
                                            tf.cast(input_length[:,tf.newaxis], tf.int32),
                                            tf.cast(phone_label_length[:,tf.newaxis], tf.int32),
                                            )

            ctc_decode_result=tf.keras.backend.ctc_decode(ctc_output,input_length)[0][0]
            ctc_decode_result=tf.cast(tf.clip_by_value(ctc_decode_result,0,self.phone_featurizer.num_classes),tf.int32)
            label_out = self.translator(tf.concat([phone_labels,tf.zeros_like(phone_labels)[:,:5]],-1),enc_output, training=True)
            ctc_out = self.translator(ctc_decode_result,enc_output, training=True)

            translate_loss=self.mask_loss(tar_label,label_out[:,:max_length])*2.+ self.mask_loss(tar_label, ctc_out[:,:max_length])
            # train_loss = tf.reduce_mean(ctc_loss+translate_loss*5.)
            train_loss=tf.nn.compute_average_loss(ctc_loss+translate_loss*2.,global_batch_size=self.global_batch_size)

            if self.is_mixed_precision:
                scaled_train_loss = self.optimizer.get_scaled_loss(train_loss)
        train_list=self.encoder.trainable_variables+self.ctc_model.trainable_variables+self.translator.trainable_variables
        if self.is_mixed_precision:
            scaled_gradients = tape.gradient(scaled_train_loss, train_list)
            gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = tape.gradient(train_loss, train_list)
        self.optimizer.apply_gradients(zip(gradients, train_list))

        ctc_acc = self.ctc_acc(phone_labels, ctc_decode_result)
        translate_acc=self.translate_acc(tar_label,ctc_out,max_length)
        self.train_metrics['ctc_acc'].update_state(ctc_acc)
        self.train_metrics["ctc_loss"].update_state(ctc_loss)
        self.train_metrics["train_loss"].update_state(train_loss)
        self.train_metrics["translate_loss"].update_state(translate_loss)
        self.train_metrics["translate_acc"].update_state(translate_acc)


    @tf.function(experimental_relax_shapes=True)
    def _eval_step(self, batch):
        features, input_length, phone_labels, phone_label_length, tar_label = batch
        max_length = tf.shape(tar_label)[1]
        enc_output = self.encoder(features, training=False)
        ctc_output = self.ctc_model(enc_output, training=False)

        ctc_output = tf.nn.softmax(ctc_output, -1)
        ctc_loss = tf.keras.backend.ctc_batch_cost(tf.cast(phone_labels, tf.int32),
                                                   tf.cast(ctc_output, tf.float32),
                                                   tf.cast(input_length[:, tf.newaxis], tf.int32),
                                                   tf.cast(phone_label_length[:, tf.newaxis], tf.int32),
                                                   )
        ctc_decode = tf.keras.backend.ctc_decode(ctc_output, input_length)[0][0]
        ctc_decode = tf.cast(tf.clip_by_value(ctc_decode, 0, self.phone_featurizer.num_classes), tf.int32)

        ctc_out = self.translator(ctc_decode, enc_output, training=False)


        translate_loss = tf.keras.losses.sparse_categorical_crossentropy(tar_label, ctc_out[:, :max_length], True)
        ctc_acc = self.ctc_acc(phone_labels, ctc_decode)
        translate_acc = self.translate_acc(tar_label, ctc_out, max_length)
        self.eval_metrics["ctc_loss"].update_state(ctc_loss)
        self.eval_metrics["translate_loss"].update_state(translate_loss)
        self.eval_metrics['ctc_acc'].update_state(ctc_acc)
        self.eval_metrics['translate_acc'].update_state(translate_acc)

    def compile(self,
                max_to_keep: int = 10):

        with self.strategy.scope():

            if not self.speech_config['streaming']:
                self.encoder = ConformerEncoder(dmodel=self.model_config['dmodel'],
                                                reduction_factor=self.model_config['reduction_factor'],
                                                num_blocks=self.model_config['num_blocks'],
                                                head_size=self.model_config['head_size'],
                                                num_heads=self.model_config['num_heads'],
                                                kernel_size=self.model_config['kernel_size'],
                                                fc_factor=self.model_config['fc_factor'],
                                                dropout=self.model_config['dropout'],
                                                add_wav_info=self.speech_config['add_wav_info'],
                                                sample_rate=self.speech_config['sample_rate'],
                                                n_mels=self.speech_config['num_feature_bins'],
                                                mel_layer_type=self.speech_config['mel_layer_type'],
                                                mel_layer_trainable=self.speech_config['mel_layer_trainable'],
                                                stride_ms=self.speech_config['stride_ms'],
                                                name="conformer_encoder", )
            else:
                self.encoder = StreamingConformerEncoder(dmodel=self.model_config['dmodel'],
                                                         reduction_factor=self.model_config['reduction_factor'],
                                                         num_blocks=self.model_config['num_blocks'],
                                                         head_size=self.model_config['head_size'],
                                                         num_heads=self.model_config['num_heads'],
                                                         kernel_size=self.model_config['kernel_size'],
                                                         fc_factor=self.model_config['fc_factor'],
                                                         dropout=self.model_config['dropout'],
                                                         add_wav_info=self.speech_config['add_wav_info'],
                                                         sample_rate=self.speech_config['sample_rate'],
                                                         n_mels=self.speech_config['num_feature_bins'],
                                                         mel_layer_type=self.speech_config['mel_layer_type'],
                                                         mel_layer_trainable=self.speech_config['mel_layer_trainable'],
                                                         stride_ms=self.speech_config['stride_ms'],
                                                         name="stream_conformer_encoder")

                self.encoder.add_chunk_size(chunk_size=int(self.speech_config['streaming_bucket']*self.speech_config['sample_rate']),
                                            mel_size=self.speech_config['num_feature_bins'],
                                            hop_size=int(self.speech_config['stride_ms'] * self.speech_config['sample_rate'] // 1000)*self.model_config['reduction_factor'])
            self.ctc_model = CTCDecoder(num_classes=self.phone_featurizer.num_classes,
                                        dmodel=self.model_config['dmodel'],
                                        num_blocks=self.model_config['ctcdecoder_num_blocks'],
                                        head_size=self.model_config['head_size'],
                                        num_heads=self.model_config['num_heads'],
                                        kernel_size=self.model_config['ctcdecoder_kernel_size'],
                                        dropout=self.model_config['ctcdecoder_dropout'],
                                        fc_factor=self.model_config['ctcdecoder_fc_factor'],
                                        )
            self.translator = Translator(inp_classes=self.phone_featurizer.num_classes,
                                         tar_classes=self.text_featurizer.num_classes,
                                         dmodel=self.model_config['dmodel'],
                                         num_blocks=self.model_config['translator_num_blocks'],
                                         head_size=self.model_config['head_size'],
                                         num_heads=self.model_config['num_heads'],
                                         kernel_size=self.model_config['translator_kernel_size'],
                                         dropout=self.model_config['translator_dropout'],
                                         fc_factor=self.model_config['translator_fc_factor'], )
            self.encoder._build()
            self.ctc_model._build()
            self.translator._build()
            try:
                self.load_checkpoint()
            except:
                logging.info('trainer resume failed')
            self.encoder.summary(line_length=100)
            self.ctc_model.summary(line_length=100)
            self.translator.summary(line_length=100)
            self.optimizer = tf.keras.optimizers.Adam(lr=self.opt_config['lr'],beta_1=self.opt_config['beta1'],
                                                      beta_2=self.opt_config['beta2'],epsilon=self.opt_config['epsilon'])
            if self.is_mixed_precision:
                self.optimizer = mixed_precision.LossScaleOptimizer(self.optimizer, "dynamic")
        self.set_progbar()
        self.max_to_keep=max_to_keep

    def fit(self, epoch=None):
        if epoch is not None:
            self.epochs = epoch
            self.train_progbar.set_description_str(
                f"[Train] [Epoch {epoch}/{self.running_config['num_epochs']}]")
        self._train_batches()
        self._check_eval_interval()

    def _train_batches(self):
        """Train model one epoch."""

        for batch in self.train_datasets:
            try:
                self.strategy.run(self._train_step, args=(batch,))
                self.steps += 1
                self.train_progbar.update(1)
                self._print_train_metrics(self.train_progbar)
                self._check_log_interval()

                if self._check_save_interval():
                    break

            except tf.errors.OutOfRangeError:
                continue

    def _eval_batches(self):
        """One epoch evaluation."""

        for metric in self.eval_metrics.keys():
            self.eval_metrics[metric].reset_states()
        n = 0
        for batch in self.eval_datasets:
            try:
                self.strategy.run(self._eval_step, args=(batch,))
            except tf.errors.OutOfRangeError:
                pass
            n += 1

            self.eval_progbar.update(1)

            self._print_eval_metrics(self.eval_progbar)
            if n >= self.eval_steps_per_epoch:
                break
        self._write_to_tensorboard(self.eval_metrics, self.steps, stage="eval")
    def save_checkpoint(self,):
        """Save checkpoint."""
        self.checkpoint_dir = os.path.join(self.running_config["outdir"], "encoder-ckpt")
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.encoder.save_weights(os.path.join(self.checkpoint_dir,'model_{}.h5'.format(self.steps)))

        if len(os.listdir(self.checkpoint_dir))>self.max_to_keep:
            files=os.listdir(self.checkpoint_dir)
            files.sort(key=lambda x:int(x.split('_')[-1].replace('.h5','')))
            os.remove(os.path.join(self.checkpoint_dir,files[0]))

        self.checkpoint_dir = os.path.join(self.running_config["outdir"], "ctc_decoder-ckpt")
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.ctc_model.save_weights(os.path.join(self.checkpoint_dir, 'model_{}.h5'.format(self.steps)))

        if len(os.listdir(self.checkpoint_dir)) > self.max_to_keep:
            files = os.listdir(self.checkpoint_dir)
            files.sort(key=lambda x: int(x.split('_')[-1].replace('.h5', '')))
            os.remove(os.path.join(self.checkpoint_dir, files[0]))

        self.checkpoint_dir = os.path.join(self.running_config["outdir"], "translator-ckpt")
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.translator.save_weights(os.path.join(self.checkpoint_dir, 'model_{}.h5'.format(self.steps)))
        self.train_progbar.set_postfix_str("Successfully Saved Checkpoint")
        if len(os.listdir(self.checkpoint_dir)) > self.max_to_keep:
            files = os.listdir(self.checkpoint_dir)
            files.sort(key=lambda x: int(x.split('_')[-1].replace('.h5', '')))
            os.remove(os.path.join(self.checkpoint_dir, files[0]))

    def load_checkpoint(self, ):
        """Load checkpoint."""

        self.checkpoint_dir = os.path.join(self.running_config["outdir"], "encoder-ckpt")
        files = os.listdir(self.checkpoint_dir)
        files.sort(key=lambda x: int(x.split('_')[-1].replace('.h5', '')))
        self.encoder.load_weights(os.path.join(self.checkpoint_dir, files[-1]))
        logging.info('encoder load at {}'.format(os.path.join(self.checkpoint_dir, files[-1])))

        self.checkpoint_dir = os.path.join(self.running_config["outdir"], "ctc_decoder-ckpt")
        files = os.listdir(self.checkpoint_dir)
        files.sort(key=lambda x: int(x.split('_')[-1].replace('.h5', '')))
        self.ctc_model.load_weights(os.path.join(self.checkpoint_dir, files[-1]))
        logging.info('ctc_model load at {}'.format(os.path.join(self.checkpoint_dir, files[-1])))

        self.checkpoint_dir = os.path.join(self.running_config["outdir"], "translator-ckpt")
        files = os.listdir(self.checkpoint_dir)
        files.sort(key=lambda x: int(x.split('_')[-1].replace('.h5', '')))
        self.translator.load_weights(os.path.join(self.checkpoint_dir, files[-1]))
        logging.info('translator load at {}'.format(os.path.join(self.checkpoint_dir, files[-1])))

        self.steps = int(files[-1].split('_')[-1].replace('.h5', ''))