import tensorflow_addons as tfa
import tensorflow as tf
from src.utils.core import red

class QAModel(tf.keras.Model): 
    def __init__(
        self, backbone, 
        dropout_rates,
        hidden_layer=None, 
        concat_start_to_end=False, 
    ): 
        super().__init__()
        self.backbone = backbone
        self.hidden_layer = hidden_layer
        if hidden_layer is not None: 
            print(red('Warning: Using Hidden Layer'))
            self.hidden_layer = tf.keras.Sequential([
                tf.keras.layers.Dropout(dropout_rates.hidden_1),
                tf.keras.layers.Dense(
                    self.hidden_layer, 
                    activation=tfa.activations.mish, 
                    kernel_initializer=self._bert_initializer(0.2),
                ), 
                tf.keras.layers.Dropout(dropout_rates.hidden_2), 
            ])
        self.start_out = tf.keras.Sequential([
            tf.keras.layers.Dropout(dropout_rates.start), 
            tf.keras.layers.Dense(1, kernel_initializer=self._bert_initializer(0.2)), 
        ], name='start_positions')
        self.end_out = tf.keras.Sequential([
            tf.keras.layers.Dropout(dropout_rates.end), 
            tf.keras.layers.Dense(1, kernel_initializer=self._bert_initializer(0.2)), 
        ], name='end_positions')
    
    def call(self, inputs, training=False):
        backbone_outputs = self.backbone(
            input_ids=inputs['input_ids'], 
            attention_mask=inputs['attention_mask'], 
            token_type_ids=inputs['token_type_ids'], 
            return_dict=True, 
        )
        sequence_outputs = backbone_outputs.last_hidden_state
        if self.hidden_layer is not None: 
            hidden_outputs = self.hidden_layer(sequence_outputs)
            sequence_outputs = tf.concat([sequence_outputs, hidden_outputs], axis=-1)
        start_logits = self.start_out(sequence_outputs)
        if self.concat_start_with_end: 
            sequence_outputs = tf.concat([sequence_outputs, start_logits], axis=-1)
        end_logits = self.end_out(sequence_outputs)
        return { 
            'start_positions': start_logits, 
            'end_positions': end_logits
        }
        
    def _bert_initializer(self, initializer_range=0.2): 
        return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)