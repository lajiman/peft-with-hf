import json
import datasets
from datasets import load_dataset, DownloadManager
from datasets.tasks import QuestionAnsweringExtractive

logger = datasets.logging.get_logger(__name__)
_DESCRIPTION = """\
Aspect-based sentiment analysis (ABSA) is a text\
analysis technique that categorizes data by aspect\
and identifies the sentiment attributed to each one. \
Aspect-based sentiment analysis can be used to analyze\
customer feedback by associating specific sentiments\
with different aspects of a product or service.\
And this is the laptop sub-set of ABSA.
"""

DIR = {
    "train": "./train.json",
    "test": "./test.json",
    "dev": "./dev.json"
}

class Laptop(datasets.GeneratorBasedBuilder):
    # BUILDER_CONFIGS = [datasets.BuilderConfig(name="plain_text", version=datasets.Version("0.0.1"))]
    # DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description = _DESCRIPTION,
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "term": datasets.Value("string"),
                    "label": datasets.Value("int32"),
                }
            ),
            supervised_keys=None,
            homepage="None",
            license="None",
            citation="None",
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": DIR["train"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": DIR["test"]}),
        ]

    def _generate_examples(self, filepath):
        logger.info("generating examples from = %s", filepath)
        key = 0
        with open(filepath, encoding="utf-8") as f:
            lap = json.load(f)
            for _,value in lap.items():
                text = value["sentence"]
                term = value["term"]
                assert value["polarity"] in ["positive", "neutral", "negative"], "polarity wrong"
                if value["polarity"] == "positive":
                    label = 0
                elif value["polarity"] == "neutral":
                    label = 1
                else:
                    label = 2
                yield key, {
                    "text": text,
                    "term": term,
                    "label": label,
                }
                key += 1


if __name__ == "__main__":
    mydata = load_dataset('./SemEval14-laptop.py')
    print(mydata)
    print(mydata["train"][0])
