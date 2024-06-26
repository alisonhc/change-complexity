import numpy as np
from classifier.predictor.level_predictor import LevelPredictor


MODEL_PATH = 'models/cefr_phrase_sent_3eps.tar.gz'
level_dict = {i:i-1 for i in range(1, 7)}
num_to_level = {0: 'A1', 1: 'A2', 2: 'B1', 3: 'B2', 4: 'C1', 5: 'C2'}

model = LevelPredictor.from_path(MODEL_PATH, 'cefr_level_predictor')

if __name__ == '__main__':
    test_sentence = "I am a test sentence."
    result = model.predict_json({'text': test_sentence})
    pred_class = np.argmax(result['probs'])
    pred_label = num_to_level[pred_class]
    print(result)
    print(pred_label)