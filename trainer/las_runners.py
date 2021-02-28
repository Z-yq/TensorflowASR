
import logging
import tensorflow as tf
import tensorflow.keras.mixed_precision.experimental as mixed_precision

from utils.speech_featurizers import SpeechFeaturizer
from utils.text_featurizers import TextFeaturizer

from trainer.base_runners import BaseTrainer



class LASTrainer(BaseTrainer):
    """ Trainer for CTC Models """

    def __init__(self,
                 speech_featurizer: SpeechFeaturizer,
                 text_featurizer: TextFeaturizer,
                 config: dict,
                 is_mixed_precision: bool = False,
                 strategy=None
                 ):
        super(LASTrainer, self).__init__(config=config,)
        self.speech_featurizer = speech_featurizer
        self.text_featurizer = text_featurizer
        self.is_mixed_precision = is_mixed_precision
        self.set_strategy(strategy)
    def set_train_metrics(self):
        lists=['classes_loss','stop_loss','ctc_loss','ctc_acc','att_acc']
        if self.config['guide_attention']:
            lists+=['alig_guide_loss']
        self.train_metrics={}
        for item in lists:
            self.train_metrics.update({
            item: tf.keras.metrics.Mean("train_{}".format(item), dtype=tf.float32)
        })

    def set_eval_metrics(self):
        lists=['classes_loss','stop_loss','ctc_loss','ctc_acc','att_acc']
        if self.config['guide_attention']:
            lists += ['alig_guide_loss']
        self.eval_metrics={}
        for item in lists:
            self.eval_metrics.update({
            item: tf.keras.metrics.Mean("eval_{}".format(item), dtype=tf.float32),
        })

    @tf.function(experimental_relax_shapes=True)
    def _train_step(self, batch):
        features,  input_length, labels, label_length,guide_matrix= batch
        ctc_label=tf.where(labels==self.text_featurizer.endid(),self.text_featurizer.pad,labels)
        max_iter=tf.shape(labels)[1]
        self.model.maxinum_iterations = max_iter
        with tf.GradientTape() as tape:


            y_pred,ctc_output, stop_token_pred, aligments = self.model([features,
                                                                input_length],
                                                                tf.expand_dims(labels, -1),
                                                                label_length,
                                                                training=True)

            ctc_loss = tf.nn.ctc_loss(ctc_label, ctc_output, label_length, input_length, False, blank_index=-1)
            ctc_loss = tf.clip_by_value(ctc_loss, 0., 1000.)
            classes_loss = self.mask_loss(labels,y_pred)
            stop_loss=self.stop_loss(labels,stop_token_pred)
            if self.config['guide_attention']:
                real_length=tf.shape(y_pred)[1]
                alig_loss=self.alig_loss(guide_matrix[:,:real_length],aligments[:,:real_length])
                train_loss=classes_loss+stop_loss+alig_loss+ctc_loss
            else:
                train_loss = classes_loss + stop_loss+ctc_loss
            train_loss = tf.nn.compute_average_loss(train_loss,
                                                    global_batch_size=self.global_batch_size)
            if self.is_mixed_precision:
                scaled_train_loss = self.optimizer.get_scaled_loss(train_loss)

        if self.is_mixed_precision:
            scaled_gradients = tape.gradient(scaled_train_loss, self.model.trainable_variables)
            gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = tape.gradient(train_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_metrics['classes_loss'].update_state(classes_loss)
        self.train_metrics["stop_loss"].update_state(stop_loss)
        self.train_metrics["ctc_loss"].update_state(ctc_loss)
        if self.config['guide_attention']:
            self.train_metrics["alig_guide_loss"].update_state(alig_loss)

        ctc_pred = tf.keras.backend.ctc_decode(tf.nn.softmax(ctc_output, -1), input_length)[0][0]
        return ctc_label, ctc_pred, y_pred


    @tf.function(experimental_relax_shapes=True)
    def _eval_step(self, batch):
        features, input_length, labels, label_length, guide_matrix = batch
        ctc_label = tf.where(labels == self.text_featurizer.endid(), self.text_featurizer.pad, labels)
        max_iter = tf.shape(labels)[1]
        self.model.maxinum_iterations=max_iter


        y_pred, ctc_output,stop_token_pred, aligments = self.model([features,
                                                        input_length],
                                                        tf.expand_dims(labels, -1),
                                                        label_length,
                                                         training=False)
        ctc_loss = tf.nn.ctc_loss(ctc_label, ctc_output, label_length, input_length, False, blank_index=-1)
        ctc_loss = tf.clip_by_value(ctc_loss, 0., 1000.)

        classes_loss = self.mask_loss(labels, y_pred)
        stop_loss = self.stop_loss(labels, stop_token_pred)
        if self.config['guide_attention']:
            real_length = tf.shape(y_pred)[1]
            alig_loss = self.alig_loss(guide_matrix[:, :real_length], aligments[:, :real_length])

        self.eval_metrics['classes_loss'].update_state(classes_loss)
        self.eval_metrics["stop_loss"].update_state(stop_loss)
        self.eval_metrics["ctc_loss"].update_state(ctc_loss)
        if self.config['guide_attention']:
            self.eval_metrics["alig_guide_loss"].update_state(alig_loss)
        ctc_pred=tf.keras.backend.ctc_decode(tf.nn.softmax(ctc_output,-1),input_length)[0][0]
        return ctc_label,ctc_pred,y_pred

    def stop_loss(self,labels,y_pred):
        y_true=tf.cast(tf.equal(labels,self.text_featurizer.pad),1.)
        loss=tf.keras.losses.binary_crossentropy(y_true,y_pred,True)
        return loss

    def mask_loss(self,y_true,y_pred):
        mask=tf.cast(tf.not_equal(y_true,self.text_featurizer.pad),1.)
        loss=tf.keras.losses.sparse_categorical_crossentropy(y_true,y_pred,True)
        mask_loss=loss*mask
        total_loss=tf.reduce_sum(mask_loss,-1)/(tf.reduce_sum(mask,-1)+1e-6)
        return total_loss
    def alig_loss(self,guide_matrix,y_pred):
        attention_masks = tf.cast(tf.math.not_equal(guide_matrix, -1.0), tf.float32)
        loss_att = tf.reduce_sum(
            tf.abs(y_pred * guide_matrix) * attention_masks,-1
        )
        loss_att /= tf.reduce_sum(attention_masks,-1)
        return tf.reduce_sum(loss_att,-1)*20.

    def ctc_acc(self, labels, y_pred):
        T1 = y_pred.shape[1]
        T2 = labels.shape[1]
        T = min([T1, T2])
        y_pred = y_pred[:, :T]
        labels = labels[:, :T]

        mask = tf.cast(tf.not_equal(labels, self.text_featurizer.pad), 1.)
        y_pred = tf.cast(y_pred, tf.float32)
        labels = tf.cast(labels, tf.float32)

        value = tf.cast(labels == y_pred, tf.float32)

        accs = tf.reduce_sum(value, -1) / (tf.reduce_sum(mask, -1) + 1e-6)
        return accs

    def att_acc(self, labels, y_pred):
        mask = tf.cast(tf.not_equal(labels, self.text_featurizer.pad), 1.)
        y_pred = tf.argmax(y_pred, -1)
        y_pred = tf.cast(y_pred, tf.float32)
        labels = tf.cast(labels, tf.float32)
        value = tf.cast(labels == y_pred, tf.float32)

        accs = tf.reduce_sum(value, -1) / (tf.reduce_sum(mask, -1) + 1e-6)
        return accs

    def compile(self, model: tf.keras.Model,
                optimizer: any,
                max_to_keep: int = 10):
        f,c=self.speech_featurizer.compute_feature_dim()
        with self.strategy.scope():
            self.model = model
            if self.model.mel_layer is not None:
                self.model._build([1, 16000,1], training=True)
            else:
                self.model._build([1, 80, f, c],training=True)
            try:
                self.load_checkpoint()
            except:
                logging.info('trainer resume failed')
            self.optimizer = tf.keras.optimizers.get(optimizer)

            if self.is_mixed_precision:
                self.optimizer = mixed_precision.LossScaleOptimizer(self.optimizer, "dynamic")

        self.set_progbar()
        # self.load_checkpoint()

    def fit(self, epoch=None):
        if epoch is not None:
            self.epochs=epoch
            self.train_progbar.set_description_str(
                f"[Train] [Epoch {epoch}/{self.config['num_epochs']}]")


        self._train_batches()

        self._check_eval_interval()

    def _train_batches(self):
        """Train model one epoch."""

        for batch in self.train_datasets:
            try:
                features, input_length, labels, label_length, guide_matrix = batch
                ctc_label,ctc_pred,y_pred=self.strategy.run(self._train_step,args=(batch,))

                ctc_acc=self.ctc_acc(ctc_label,ctc_pred)
                att_acc=self.att_acc(labels,y_pred)
                self.train_metrics['ctc_acc'].update_state(ctc_acc)
                self.train_metrics['att_acc'].update_state(att_acc)
                self.steps+=1
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
        n=0
        for batch in self.eval_datasets:
            try:
                features, input_length, labels, label_length, guide_matrix = batch
                ctc_label,ctc_pred,y_pred=self.strategy.run(self._eval_step,args=(batch,))
                ctc_acc = self.ctc_acc(ctc_label, ctc_pred)
                att_acc = self.att_acc(labels, y_pred)
                self.eval_metrics['ctc_acc'].update_state(ctc_acc)
                self.eval_metrics['att_acc'].update_state(att_acc)

            except tf.errors.OutOfRangeError:

                pass
            n+=1

            # Update steps
            self.eval_progbar.update(1)


            # Print eval info to progress bar
            self._print_eval_metrics(self.eval_progbar)
            if n>=self.eval_steps_per_epoch:
                break
        self._write_to_tensorboard(self.eval_metrics, self.steps, stage="eval")