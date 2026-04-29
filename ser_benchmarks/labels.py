"""Label maps used by the dataset manifest builders."""

IEMOCAP_4CLS = {
    "ang": "angry",
    "sad": "sad",
    "neu": "neutral",
    "hap": "happy",
    "exc": "happy",
}

IEMOCAP_5CLS = {
    "ang": "angry",
    "sad": "sad",
    "neu": "neutral",
    "hap": "happy",
    "exc": "excited",
}

MELD_EMOTION_MAP = {
    "anger": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "joy": "happy",
    "neutral": "neutral",
    "sadness": "sad",
    "surprise": "surprise",
}

ESD_EMOTION_MAP = {
    "Angry": "angry",
    "Happy": "happy",
    "Neutral": "neutral",
    "Sad": "sad",
    "Surprise": "surprise",
    "Evaluation Set": "eval",
    "Test Set": "test",
    "Training Set": "train",
}

MSP_LABEL_MAP = {
    "A": "angry",
    "H": "happy",
    "S": "sad",
    "N": "neutral",
}
