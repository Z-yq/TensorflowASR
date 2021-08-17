
import os
from tqdm import tqdm
from colorama import Fore
import tensorflow as tf
from utils.tools import preprocess_paths



class BaseRunner():
    """ Customized runner module for all models """

    def __init__(self, config: dict):
        """
        running_config:
            batch_size: 8
            num_epochs:          20
            outdir:              ...
            log_interval_steps:  200
            eval_interval_steps: 200
            save_interval_steps: 200
        """
        self.config = config
        self.config["outdir"] = preprocess_paths(self.config["outdir"])
        # Writers
        self.train_writer = tf.summary.create_file_writer(
            os.path.join(config["outdir"], "tensorboard", "train")
        )
        self.eval_writer = tf.summary.create_file_writer(
            os.path.join(config["outdir"], "tensorboard", "eval")
        )

    def _write_to_tensorboard(self,
                              list_metrics: dict,
                              step: any,
                              stage: str = "train"):
        """Write variables to tensorboard."""
        assert stage in ["train", "eval"]
        if stage == "train":
            writer = self.train_writer
        else:
            writer = self.eval_writer
        with writer.as_default():
            for key, value in list_metrics.items():
                tf.summary.scalar(key, value.result(), step=step)
                writer.flush()


class BaseTrainer(BaseRunner):
    """Customized trainer module for all models."""

    def __init__(self,
                 config: dict,
                 ):
        """
        Args:
            config: the 'learning_config' part in YAML config file
        """
        # Configurations
        super(BaseTrainer, self).__init__(config)
        # Steps and Epochs start from 0
        self.steps = 0 # Step must be int64 to use tf.summary
        self.epochs = 0
        self.train_steps_per_epoch = config['train_steps_per_batches']
        self.eval_steps_per_epoch = config['eval_steps_per_batches']
        # Dataset
        self.train_data_loader = None
        self.eval_data_loader = None
        self.total_train_steps=None
        self.model=None
        self.set_train_metrics()
        self.set_eval_metrics()

    def set_strategy(self, strategy=None):
        if not strategy:
            self.strategy=tf.distribute.MirroredStrategy()
        else:
            self.strategy = strategy
        self.global_batch_size = self.config["batch_size"] * self.strategy.num_replicas_in_sync

    def set_total_train_steps(self,value):
        # if self.train_steps_per_epoch is None:
        #     return None
        self.total_train_steps=value
        # return self.config["train_steps_per_epoch"]
    def set_progbar(self):
        self.train_progbar = tqdm(
            initial=self.steps, unit="batch", total=self.total_train_steps,
            position=0, leave=True,
            bar_format="{desc} |%s{bar:20}%s{r_bar}" % (Fore.GREEN, Fore.RESET),
            desc="[Train]"
        )
        self.eval_progbar = tqdm(
            initial=0, total=self.eval_steps_per_epoch, unit="batch",
            position=0, leave=True,
            bar_format="{desc} |%s{bar:20}%s{r_bar}" % (Fore.BLUE, Fore.RESET),
            desc=f"[Eval] [Step {self.steps}]"
        )

    # -------------------------------- GET SET -------------------------------------

    # @abc.abstractmethod
    def set_train_metrics(self):
        self.train_metrics = {}
        raise NotImplementedError()

    # @abc.abstractmethod
    def set_eval_metrics(self):
        self.eval_metrics = {}
        raise NotImplementedError()


    # -------------------------------- CHECKPOINTS -------------------------------------



    def save_checkpoint(self,max_save=10):
        """Save checkpoint."""
        self.checkpoint_dir = os.path.join(self.config["outdir"], "checkpoints")
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.model.save_weights(os.path.join(self.checkpoint_dir,'model_{}.h5'.format(self.steps)))
        self.train_progbar.set_postfix_str("Successfully Saved Checkpoint")
        if len(os.listdir(self.checkpoint_dir))>max_save:
            files=os.listdir(self.checkpoint_dir)
            files.sort(key=lambda x:int(x.split('_')[-1].replace('.h5','')))
            os.remove(os.path.join(self.checkpoint_dir,files[0]))

    # -------------------------------- RUNNING -------------------------------------

    def _finished(self,):

        return self.steps >= (self.total_train_steps)

    def load_checkpoint(self,):
        """Load checkpoint."""

        self.checkpoint_dir = os.path.join(self.config["outdir"], "checkpoints")
        files = os.listdir(self.checkpoint_dir)
        files.sort(key=lambda x: int(x.split('_')[-1].replace('.h5', '')))
        self.model.load_weights(os.path.join(self.checkpoint_dir, files[-1]))
        self.steps= int(files[-1].split('_')[-1].replace('.h5', ''))
    def _train_batches(self):
        """Train model one epoch."""

        for batch in self.train_datasets:
            try:
                self.strategy.run(self._train_step,args=(batch,))
                self.steps+=1
                self.train_progbar.update(1)
                self._print_train_metrics(self.train_progbar)
                self._check_log_interval()

                if self._check_save_interval():
                    break

            except tf.errors.OutOfRangeError:
                continue

    def set_datasets(self,train,eval):
        self.train_datasets=self.strategy.experimental_distribute_dataset(train)

        self.eval_datasets=self.strategy.experimental_distribute_dataset(eval)


    def _train_step(self, batch):
        """ One step training. Does not return anything"""
        raise NotImplementedError()

    def _eval_batches(self):
        """One epoch evaluation."""

        for metric in self.eval_metrics.keys():
            self.eval_metrics[metric].reset_states()
        n=0
        for batch in self.eval_datasets:
            try:
                self.strategy.run(self._eval_step,args=(batch,))

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

    def _eval_step(self, batch):
        """One eval step. Does not return anything"""
        raise NotImplementedError()

    def compile(self, *args, **kwargs):
        """ Function to initialize models and optimizers """
        raise NotImplementedError()

    def fit(self, *args, **kwargs):
        """ Function run start training, including executing "run" func """
        raise NotImplementedError()

    # -------------------------------- LOGGING -------------------------------------

    def _check_log_interval(self):
        """Save log interval."""
        if self.steps % self.config["log_interval_steps"] == 0:
            self._write_to_tensorboard(self.train_metrics, self.steps, stage="train")
            """Reset train metrics after save it to tensorboard."""
            for metric in self.train_metrics.keys():
                self.train_metrics[metric].reset_states()

    def _check_save_interval(self):
        """Save log interval."""
        if self.steps % self.config["save_interval_steps"] == 0:
            self.save_checkpoint()
            return True
        return False

    def _check_eval_interval(self):
        """Save log interval."""
        if self.steps % self.config["eval_interval_steps"] == 0:
            self._eval_batches()

    # -------------------------------- UTILS -------------------------------------

    def _print_train_metrics(self, progbar):
        result_dict = {}
        for key, value in self.train_metrics.items():
            result_dict[f"{key}"] = str(round(float(value.result().numpy()),3))
        progbar.set_postfix(result_dict)

    def _print_eval_metrics(self, progbar):
        result_dict = {}
        for key, value in self.eval_metrics.items():
            result_dict[f"{key}"] = str(round(float(value.result().numpy()),3))
        progbar.set_postfix(result_dict)

    # -------------------------------- END -------------------------------------

