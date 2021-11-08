
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

        self.pad=0
        self.stop=-1

    def startid(self):
        return self.token_to_index['<S>']
    def endid(self):
        return self.token_to_index['</S>']

    def extract(self, tokens):
        feats = [self.token_to_index[token] for token in tokens]
        return feats


    def iextract(self, feat):
        """
        Args:
            feat: tf.Tensor with dim [B, None]

        Returns:
            transcripts: tf.Tensor of dtype tf.string with dim [B]
        """
        # return tf.map_fn(lambda x: tf.numpy_function(self._idx_to_char, [x], tf.string),
        #                  feat, dtype=tf.string)
        tokens=[self.index_to_token[index] for index in feat]
        return tokens


