
import logging
import tensorflow as tf
import tensorflow.keras.mixed_precision.experimental as mixed_precision

from utils.speech_featurizers import SpeechFeaturizer
from utils.text_featurizers import TextFeaturizer

from trainer.base_runners import BaseTrainer



class MultiTaskLASTrainer(BaseTrainer):
    """ Trainer for CTC Models """

    def __init__(self,
                 speech_featurizer: SpeechFeaturizer,
                 text_featurizer: TextFeaturizer,
                 config: dict,
                 is_mixed_precision: bool = False,
                 strategy=None
                 ):
        super(MultiTaskLASTrainer, self).__init__(config=config,)
        self.speech_featurizer = speech_featurizer
        self.text_featurizer = text_featurizer
        self.is_mixed_precision = is_mixed_precision
        self.set_strategy(strategy)
    def set_train_metrics(self):
        lists=['classes_loss','stop_loss','ctc1_loss','ctc2_loss','ctc3_loss','feature_map_loss']
        if self.config['guide_attention']:
            lists+=['alig_guide_loss']
        self.train_metrics={}
        for item in lists:
            self.train_metrics.update({
            item: tf.keras.metrics.Mean("train_{}".format(item), dtype=tf.float32)
        })

    def set_eval_metrics(self):
        lists = ['classes_loss', 'stop_loss','ctc1_loss','ctc2_loss','ctc3_loss','feature_map_loss']
        if self.config['guide_attention']:
            lists += ['alig_guide_loss']
        self.eval_metrics={}
        for item in lists:
            self.eval_metrics.update({
            item: tf.keras.metrics.Mean("eval_{}".format(item), dtype=tf.float32),
        })

    def bert_feature_loss(self, real, pred):
        mask = tf.cast(tf.not_equal(real, -10.), 1.)
        loss = tf.square(real - pred)
        loss *= mask
        return tf.reduce_mean(tf.reduce_sum(loss, -1) / (tf.reduce_sum(mask, -1) + 1e-6), -1, True)
    def classes_acc(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        accs = tf.keras.metrics.sparse_categorical_accuracy(real,pred)

        mask = tf.cast(mask, dtype=accs.dtype)
        accs *= mask
        final=tf.reduce_sum(accs,-1)/tf.reduce_sum(mask,-1)

        return tf.reduce_mean(final)
    @tf.function(experimental_relax_shapes=True)
    def _train_step(self, batch):
        x, wavs, bert_feature, input_length, words_label, words_label_length, phone_label, phone_label_length, py_label, py_label_length, txt_label, txt_label_length, guide_matrix= batch

        with tf.GradientTape() as tape:
            ctc1_output, ctc2_output, ctc3_output, final_decoded, bert_output, stop_token_pred, alignments = self.model(x,
            input_length,
            bert_feature,
            txt_label_length,
             training=True)
            ctc1_output = tf.nn.softmax(ctc1_output, -1)
            ctc2_output = tf.nn.softmax(ctc2_output, -1)
            ctc3_output = tf.nn.softmax(ctc3_output, -1)

            tape.watch(final_decoded)
            classes_loss = self.mask_loss(txt_label,final_decoded)
            tape.watch(stop_token_pred)
            stop_loss=self.stop_loss(txt_label,stop_token_pred)
            feature_map_loss = self.bert_feature_loss(bert_feature, bert_output)
            ctc1_loss=tf.keras.backend.ctc_batch_cost(words_label,ctc1_output,input_length[:,tf.newaxis],words_label_length[:,tf.newaxis])

            ctc2_loss=tf.keras.backend.ctc_batch_cost(phone_label,ctc2_output,input_length[:,tf.newaxis],phone_label_length[:,tf.newaxis])

            ctc3_loss=tf.keras.backend.ctc_batch_cost(py_label,ctc3_output,input_length[:,tf.newaxis],py_label_length[:,tf.newaxis])
            if self.config['guide_attention']:
                real_length=tf.shape(final_decoded)[1]
                tape.watch(alignments)
                alig_loss=self.alig_loss(guide_matrix[:,:real_length],alignments[:,:real_length])
                train_loss=classes_loss+stop_loss+alig_loss+feature_map_loss+ctc1_loss+ctc2_loss+ctc3_loss
            else:
                train_loss = classes_loss + stop_loss+feature_map_loss+ctc1_loss+ctc2_loss+ctc3_loss
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
        self.train_metrics["ctc1_loss"].update_state(ctc1_loss)
        self.train_metrics["ctc2_loss"].update_state(ctc2_loss)
        self.train_metrics["ctc3_loss"].update_state(ctc3_loss)
        self.train_metrics["feature_map_loss"].update_state(feature_map_loss)
        if self.config['guide_attention']:
            self.train_metrics["alig_guide_loss"].update_state(alig_loss)




    @tf.function(experimental_relax_shapes=True)
    def _eval_step(self, batch):
        x, wavs, bert_feature, input_length, words_label, words_label_length, phone_label, phone_label_length, py_label, py_label_length, txt_label, txt_label_length, guide_matrix = batch

        ctc1_output, ctc2_output, ctc3_output, final_decoded, bert_output, stop_token_pred, alignments = self.model(x,
                                                                                                                    input_length,
                                                                                                                    bert_feature,
                                                                                                                    txt_label_length,
                                                                                                                    training=True)

        classes_loss = self.mask_loss(txt_label, final_decoded)
        stop_loss = self.stop_loss(txt_label, stop_token_pred)
        feature_map_loss = self.bert_feature_loss(bert_feature, bert_output)
        ctc1_loss = tf.nn.ctc_loss(words_label, ctc1_output, words_label_length, input_length, False,
                                   blank_index=self.text_featurizer.blank)
        ctc2_loss = tf.nn.ctc_loss(phone_label, ctc2_output, phone_label_length, input_length, False,
                                   blank_index=self.text_featurizer.blank)
        ctc3_loss = tf.nn.ctc_loss(py_label, ctc3_output, py_label_length, input_length, False,
                                   blank_index=self.text_featurizer.blank)
        if self.config['guide_attention']:
            real_length = tf.shape(final_decoded)[1]
            alig_loss = self.alig_loss(guide_matrix[:, :real_length], alignments[:, :real_length])

        self.train_metrics['classes_loss'].update_state(classes_loss)
        self.train_metrics["stop_loss"].update_state(stop_loss)
        self.train_metrics["ctc1_loss"].update_state(ctc1_loss)
        self.train_metrics["ctc2_loss"].update_state(ctc2_loss)
        self.train_metrics["ctc3_loss"].update_state(ctc3_loss)
        self.train_metrics["feature_map_loss"].update_state(feature_map_loss)
        if self.config['guide_attention']:
            self.train_metrics["alig_guide_loss"].update_state(alig_loss)


    def stop_loss(self,labels,y_pred):
        y_true = tf.cast(tf.equal(labels, self.text_featurizer.blank), 1.)
        loss=tf.keras.losses.binary_crossentropy(y_true,y_pred,True)
        return loss

    def mask_loss(self,y_true,y_pred):
        mask = tf.cast(tf.not_equal(y_true, self.text_featurizer.blank), 1.)
        loss=tf.keras.losses.sparse_categorical_crossentropy(y_true,y_pred,True)
        mask_loss=loss*mask
        total_loss=tf.reduce_mean(tf.reduce_sum(mask_loss,-1)/(tf.reduce_sum(mask,-1)+1e-6),-1)
        return tf.reduce_sum(total_loss)
    def alig_loss(self,guide_matrix,y_pred):
        attention_masks = tf.cast(tf.math.not_equal(guide_matrix, -1.0), tf.float32)
        loss_att = tf.reduce_sum(
            tf.abs(y_pred * guide_matrix) * attention_masks,-1
        )
        loss_att /= tf.reduce_sum(attention_masks,-1)
        return tf.reduce_sum(loss_att)
    def compile(self, model: tf.keras.Model,
                optimizer: any,
                max_to_keep: int = 10):
        f,c=self.speech_featurizer.compute_feature_dim()
        with self.strategy.scope():
            self.model = model

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
    def _train_batches(self):
        """Train model one epoch."""

        for idx,batch in enumerate(self.train_datasets):
            try:
                self.strategy.run(self._train_step,args=(batch,))
                self.steps+=1
                self.train_progbar.update(1)
                self._print_train_metrics(self.train_progbar)
                self._check_log_interval()
                self._check_save_interval()

            except tf.errors.OutOfRangeError:
                continue
            if idx>self.train_steps_per_epoch:
                break
    def _eval_batches(self):
        """One epoch evaluation."""

        for metric in self.eval_metrics.keys():
            self.eval_metrics[metric].reset_states()

        for idx, batch in enumerate(self.eval_datasets):
            try:
                self.strategy.run(self._eval_step,args=(batch,))

            except tf.errors.OutOfRangeError:

                pass

            # Update steps
            self.eval_progbar.update(1)


            # Print eval info to progress bar
            self._print_eval_metrics(self.eval_progbar)
            if idx >=self.eval_steps_per_epoch:
                break
        self._write_to_tensorboard(self.eval_metrics, self.steps, stage="eval")
    def fit(self, epoch=None):
        if epoch is not None:
            self.epochs=epoch
            self.train_progbar.set_description_str(
                f"[Train] [Epoch {epoch}/{self.config['num_epochs']}]")


        self._train_batches()

        self._check_eval_interval()

