
import codecs
import tensorflow as tf
from utils.tools import preprocess_paths


class TextFeaturizer:
    """
    Extract text feature based on char-level granularity.
    By looking up the vocabulary table, each line of transcript will be
    converted to a sequence of integer indexes.
    """

    def __init__(self, decoder_config: dict,show=False):
        """
        decoder_config = {
            "vocabulary": str,
            "blank_at_zero": bool,
            "beam_width": int,
            "lm_config": {
                ...
            }
        }
        """
        self.decoder_config = decoder_config

        self.decoder_config["vocabulary"] = preprocess_paths(self.decoder_config["vocabulary"])

        self.scorer = None

        self.num_classes = 0
        lines = []
        with codecs.open(self.decoder_config["vocabulary"], "r", "utf-8") as fin:
            lines.extend(fin.readlines())
        if show:
            print('load token at {}'.format(self.decoder_config['vocabulary']))
        self.token_to_index = {}
        self.index_to_token = {}
        self.vocab_array = []
        self.tf_vocab_array = tf.constant([], dtype=tf.string)
        self.index_to_unicode_points = tf.constant([], dtype=tf.int32)
        index = 0
        if self.decoder_config["blank_at_zero"]:
            self.blank = 0
            index = 1
            self.tf_vocab_array = tf.concat([self.tf_vocab_array, [""]], axis=0)
            self.index_to_unicode_points = tf.concat(
                [self.index_to_unicode_points, [0]], axis=0)
        for line in lines:
            line = line.strip()  # Strip the '\n' char
            # Skip comment line, empty line
            if line.startswith("#") or not line or line == "\n":
                continue
            self.token_to_index[line] = index
            self.index_to_token[index] = line
            self.vocab_array.append(line)
            self.tf_vocab_array = tf.concat([self.tf_vocab_array, [line]], axis=0)
            upoint = tf.strings.unicode_decode(line, "UTF-8")
            self.index_to_unicode_points = tf.concat(
                [self.index_to_unicode_points, upoint], axis=0)
            index += 1
        self.num_classes = index
        if not self.decoder_config["blank_at_zero"]:
            self.blank = index
            self.num_classes += 1
            self.tf_vocab_array = tf.concat([self.tf_vocab_array, [""]], axis=0)
            self.index_to_unicode_points = tf.concat(
                [self.index_to_unicode_points, [0]], axis=0)
        if self.decoder_config['model_type']=='Transducer':
            self.stop=self.endid()
            self.pad=self.blank
            self.start=self.startid()
        elif self.decoder_config['model_type'] == 'LAS':
            self.stop = self.endid()
            self.pad = 0
            self.start = self.startid()
        elif self.decoder_config['model_type']=='LM':
            self.stop = self.endid()
            self.pad = 0
            self.start = self.startid()
        else:
            self.pad=0
            self.stop=-1
    def add_scorer(self, scorer: any = None):
        """ Add scorer to this instance, scorer can use decoder_config property """
        self.scorer = scorer
    def startid(self):
        return self.token_to_index['S']
    def endid(self):
        return self.token_to_index['/S']
    def prepand_blank(self, text: tf.Tensor) -> tf.Tensor:
        """ Prepand blank index for transducer models """
        return tf.concat([[self.blank], text], axis=0)

    def extract(self, tokens):
        """
        Convert string to a list of integers
        Args:
            text: string (sequence of characters)

        Returns:
            sequence of ints in tf.Tensor
        """
        # new_tokens = []
        # for tok in tokens:
        #     if tok in self.vocab_array:
        #         new_tokens.append(tok)
        # tokens = new_tokens
        if self.decoder_config['model_type']=='CTC':
            feats = [self.token_to_index[token] for token in tokens]
        elif self.decoder_config['model_type']=='LAS':
            feats = [self.token_to_index[token] for token in tokens] + [self.stop]
        else:
            feats=[self.start]+[self.token_to_index[token] for token in tokens]
        return feats

    @tf.function
    def iextract(self, feat):
        """
        Args:
            feat: tf.Tensor with dim [B, None]

        Returns:
            transcripts: tf.Tensor of dtype tf.string with dim [B]
        """
        # return tf.map_fn(lambda x: tf.numpy_function(self._idx_to_char, [x], tf.string),
        #                  feat, dtype=tf.string)
        with tf.name_scope("invert_text_extraction"):
            minus_one = -1 * tf.ones_like(feat, dtype=tf.int32)
            blank_like = self.blank * tf.ones_like(feat, dtype=tf.int32)
            feat = tf.where(feat == minus_one, blank_like, feat)
            # print(feat)
            return tf.map_fn(self._idx_to_char, feat, dtype=tf.string)


    def _idx_to_char(self, arr: tf.Tensor) -> tf.Tensor:
        transcript = tf.constant("", dtype=tf.string)
        for i in arr:
            transcript = tf.strings.join([transcript, self.tf_vocab_array[i]],separator=' ')
        return transcript

        # arr = arr[arr != self.blank]
        # arr = arr[arr != -1]
        # return "".join([self.index_to_token[i] for i in arr])

    @tf.function(
        experimental_relax_shapes=True,
        input_signature=[
            tf.TensorSpec([None, None], dtype=tf.int32)
        ]
    )
    def index2upoints(self, feat: tf.Tensor) -> tf.Tensor:
        """
        Transform Predicted Classes to Unicode Code Points (mainly for using tflite)
        TFLite Map_fn Issue: https://github.com/tensorflow/tensorflow/issues/40221
        Only use in tf-nightly
        Args:
            feat: tf.Tensor of Classes in shape [B, None]

        Returns:
            unicode code points transcript with dtype tf.int32 and shap [B, None]
        """
        with tf.name_scope("index_to_unicode_points"):
            def map_fn(arr):
                def sub_map_fn(index):
                    return self.index_to_unicode_points[index]
                return tf.map_fn(sub_map_fn, arr, dtype=tf.int32
                                 )
            # filter -1 value to avoid outofrange
            minus_one = -1 * tf.ones_like(feat, dtype=tf.int32)
            blank_like = self.blank * tf.ones_like(feat, dtype=tf.int32)
            feat = tf.where(feat == minus_one, blank_like, feat)
            return tf.map_fn(map_fn, feat, dtype=tf.int32,
                           )
if __name__ == '__main__':
    from utils.user_config import UserConfig
    import pypinyin
    import numpy as np
    config=UserConfig('../config.yml','../config.yml',False)
    print(config)
    test=TextFeaturizer(config['decoder_config'])
    print(test.num_classes,test.vocab_array)
    # print(test.extract(pypinyin.lazy_pinyin('我爱你',1)))
    print(test.iextract(tf.constant(np.random.random([4,test.num_classes]))))
