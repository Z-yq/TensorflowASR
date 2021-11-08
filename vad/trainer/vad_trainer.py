import logging
import os
import tensorflow as tf
import tensorflow.keras.mixed_precision.experimental as mixed_precision
from vad.utils.stft import TFMultiResolutionSTFT
from vad.models.vad_model import CNN_Online_VAD,CNN_Offline_VAD
from vad.trainer.base_trainer import BaseTrainer


class VADTrainer(BaseTrainer):
    """ Trainer for CTC Models """

    def __init__(self,

                 config: dict,
                 is_mixed_precision: bool = False,
                 strategy=None
                 ):
        super(VADTrainer, self).__init__(config=config['running_config'], )
        self.config = config
        self.speech_config = config['speech_config']
        self.model_config = config['model_config']
        self.opt_config=config['optimizer_config']
        self.is_mixed_precision = is_mixed_precision
        self.set_strategy(strategy)

    def set_train_metrics(self):
        self.train_metrics = {
            "vad_loss": tf.keras.metrics.Mean("vad_loss", dtype=tf.float32),
            "wav_loss": tf.keras.metrics.Mean("wav_loss", dtype=tf.float32),
            "vad_acc": tf.keras.metrics.Mean("vad_acc", dtype=tf.float32),
        }

    def set_eval_metrics(self):
        self.eval_metrics = {
            "vad_loss": tf.keras.metrics.Mean("vad_loss", dtype=tf.float32),
            "wav_loss": tf.keras.metrics.Mean("wav_loss", dtype=tf.float32),
            "vad_acc": tf.keras.metrics.Mean("vad_acc", dtype=tf.float32),

        }

    def mask_loss(self,y,pred):
        loss=tf.losses.binary_crossentropy(y,pred,True)
        one=tf.squeeze(y,-1)
        zero=tf.where(one==0.,1.,0.)
        one_loss=tf.reduce_sum(loss*one)/(tf.reduce_sum(one)+1e-6)
        zero_loss=tf.reduce_sum(loss*zero)/(tf.reduce_sum(zero)+1e-6)
        return one_loss,zero_loss

    @tf.function(experimental_relax_shapes=True)
    def _train_step(self, batch):
        x,vad_label,wav_label= batch
        with tf.GradientTape() as tape:
            g_out1, g_out2 = self.model(x,training=True)
            one, zero = self.mask_loss(vad_label, g_out1)
            sftf_loss = self.stft_loss.call(wav_label, g_out2)
            train_loss = tf.nn.compute_average_loss( (one + zero) * 10. + sftf_loss,
                                                    global_batch_size=self.global_batch_size)
            if self.is_mixed_precision:
                scaled_train_loss = self.optimizer.get_scaled_loss(train_loss)
        train_list=self.model.trainable_variables
        if self.is_mixed_precision:
            scaled_gradients = tape.gradient(scaled_train_loss, train_list)
            gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = tape.gradient(train_loss, train_list)
        self.optimizer.apply_gradients(zip(gradients, train_list))
        acc = tf.metrics.binary_accuracy(vad_label, g_out1)
        self.train_metrics['vad_loss'].update_state(one+zero)
        self.train_metrics["wav_loss"].update_state(sftf_loss)
        self.train_metrics["vad_acc"].update_state(acc)



    @tf.function(experimental_relax_shapes=True)
    def _eval_step(self, batch):
        x,vad_label,wav_label= batch
        g_out1, g_out2 = self.model(x, training=False)
        one, zero = self.mask_loss(vad_label, g_out1)
        sftf_loss = self.stft_loss.call(wav_label, g_out2)
        acc = tf.metrics.binary_accuracy(vad_label, g_out1)
        self.eval_metrics["vad_loss"].update_state(one+zero)
        self.eval_metrics["wav_loss"].update_state(sftf_loss)
        self.eval_metrics['vad_acc'].update_state(acc)


    def compile(self,
                max_to_keep: int = 10):

        with self.strategy.scope():
            if self.model_config['streaming']:
                self.model = CNN_Online_VAD(self.model_config['dmodel'], name=self.model_config['name'])
            else:
                self.model = CNN_Offline_VAD(self.model_config['dmodel'], name=self.model_config['name'])

            self.model._build()
            self.stft_loss=TFMultiResolutionSTFT(batch=self.running_config['batch_size'])
            try:
                self.load_checkpoint()
            except:
                logging.info('trainer resume failed')
            self.model.summary()
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
                x,y,y2=batch
                if self.model_config['streaming'] :
                    if tf.random.uniform(1)>0.5:
                        try:
                            V=x.shape()[-1]
                            x=tf.reshape(x,[-1,self.model_config['streaming_min_frame'],V])
                            V = y.shape()[-1]
                            y=tf.reshape(y,[-1,self.model_config['streaming_min_frame'],V])
                            V = y2.shape()[-1]
                            y2 = tf.reshape(y2, [-1, self.model_config['streaming_min_frame'], V])
                        except:
                            pass
                self.strategy.run(self._train_step, args=([x,y,y2],))
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

