from operator import concat
import transformers 
import tensorflow as tf

class QAModel(tf.keras.model.Model): 
    def __init__(
        self, 
        backbone, 
        loss_weights={'start': 1, 'negative': 1}, 
        dropouts={'start': 0, 'end': 0}, 
        concat_embeddings=True, 
    ): 
        self.backbone = backbone 
        self.start_weight = loss_weights.start
        self.negative_weight = loss_weights.negative 
        self.concat_embeddings = concat_embeddings
        
        print('loss weight: ', loss_weights)
        print('dropout: ', dropouts)
        self.start_out = tf.keras.Sequential([
            tf.keras.layers.Dropout(dropouts['start']), 
            tf.keras.layers.Dense(1, kernel_initializer= self._bert_initializer(0.2)), 
        ], name='start',)
        self.end_out = tf.keras.Sequential([
            tf.keras.layers.Dropout(dropouts['end']), 
            tf.keras.layers.Dense(1, kernel_initializer= self._bert_initializer(0.2)), 
        ], name='end',)
    
    
    def call(
        self, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds, 
        head_mask, inputs_emebds, output_attentions, output_hidden_states, 
        return_dict, start_positions, end_positions, training, 
        **kwargs
    ): 
        inputs = transformers.modelling_tf_utils.input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            start_positions=start_positions,
            end_positions=end_positions,
            training=training,
            kwargs_call=kwargs,
        )
        outputs = self.rembert(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            position_ids=inputs["position_ids"],
            head_mask=inputs["head_mask"],
            inputs_embeds=inputs["inputs_embeds"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )
        sequence_output = outputs[0]
    


























import transformers
import tensorflow as tf


    

class TFQARembert(transformers.TFRemBertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):  
        super().__init__(config, *inputs, **kwargs)
        self.rembert = transformers.TFRembertMainLayer(config, add_pooling_layer=False, name='rembert')
        self.start_out = tf.keras.layers.Dense(
            1, kernel_initializer=self._bert_initializer(config.initializer_range), name='start'
        )
        self.end_out = tf.keras.layers.Dense(
            1, kernel_initializer=self._bert_initializer(config.initializer_range), name='end'
        )
        


        start_logits = self.start_out()
        
        
        
        
        sequence_output = outputs[0]
        logits = self.qa_outputs(inputs=sequence_output)
        start_logits, end_logits = tf.split(value=logits, num_or_size_splits=2, axis=-1)
        start_logits = tf.squeeze(input=start_logits, axis=-1)
        end_logits = tf.squeeze(input=end_logits, axis=-1)
        loss = None

        if inputs["start_positions"] is not None and inputs["end_positions"] is not None:
            labels = {"start_position": inputs["start_positions"]}
            labels["end_position"] = inputs["end_positions"]
            loss = self.compute_loss(labels=labels, logits=(start_logits, end_logits))

        if not inputs["return_dict"]:
            output = (start_logits, end_logits) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TFQuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def serving_output(self, output: TFQuestionAnsweringModelOutput) -> TFQuestionAnsweringModelOutput:
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return TFQuestionAnsweringModelOutput(
            start_logits=output.start_logits, end_logits=output.end_logits, hidden_states=hs, attentions=attns
        )
