from allennlp.data import DatasetReader
from allennlp.data.fields import TextField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WhitespaceTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from typing import Iterable, Dict
import jsonlines
import sys
MULTI_LABEL_TO_INDEX = {'A1': 0, 'A2': 1, 'B1': 2, 'B2': 3, 'C1': 4, 'C2': 5}

@DatasetReader.register('example_reader')
class ExampleReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer,
                 text_token_indexers: Dict[str, TokenIndexer], to_index=6,
                 **kwargs):
        print('initialized')
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.text_token_indexers = text_token_indexers
        if to_index == 6:
            self.to_index = MULTI_LABEL_TO_INDEX

    def _read(self, file_path: str) -> Iterable[Instance]:
        with jsonlines.open(file_path) as f:
            for example in f:
                instance = self.example_to_instance(
                    example['text'], example.get('label', None))
                yield instance

    def example_to_instance(self, text: str, label: str = None) -> Instance:
        tokens = self.tokenizer.tokenize(text)
        text_field = TextField(tokens, self.text_token_indexers)
        fields = {'text': text_field}
        if label is not None:
            label_field = LabelField(self.to_index[label], skip_indexing=True)
            fields['label'] = label_field
        instance = Instance(fields)
        return instance


def main():
    file_path = sys.argv[1]
    max_len = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    example_reader = ExampleReader(
        WhitespaceTokenizer(),
        {'tokens': SingleIdTokenIndexer()},
        max_len,
    )
    cnt = 0
    for instance in example_reader.read(file_path):
        print(instance)
        cnt += 1
        if cnt == 10:
            break


if __name__ == '__main__':
    main()


