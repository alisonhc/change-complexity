from allennlp.data import Instance
from allennlp.common.util import JsonDict
from allennlp.predictors import Predictor
from typing import List, Dict, Any
from overrides import overrides


@Predictor.register('cefr_level_predictor')
class LevelPredictor(Predictor):
    @overrides
    def predict_json(self, inputs: JsonDict) -> Dict[str, Any]:
        probs = self.predict_probs(inputs)
        return {'text': inputs['text'], 'probs': probs}

    def predict_probs(self, inputs: JsonDict):
        """
        Args:
            inputs: a dictionary containing two keys
                (1) word (optional)
                (2) definition: need to be tokenized

        Returns:
            def_embeds: definition embeddings, a list consists of 300 floating points
        """
        instance = self._json_to_instance(inputs)
        output_dict = self.predict_instance(instance)
        probs = output_dict['probs']  
        return probs

    @overrides
    def predict_batch_json(self, inputs: List[JsonDict]) -> List[Dict[str, Any]]:
        instances = self._batch_json_to_instances(inputs)
        output_dicts = self.predict_batch_instance(instances)
        results = []
        for inp, od in zip(inputs, output_dicts):
            results.append(
                {'text': inp['text'], 'probs': od['probs']}) 
        return results

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        text = json_dict['text']
        return self._dataset_reader.example_to_instance(text=text, label=None)