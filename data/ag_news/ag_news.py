import datasets
import pandas as pd
from datasets import load_dataset, DownloadManager
from datasets.tasks import QuestionAnsweringExtractive

logger = datasets.logging.get_logger(__name__)
_DESCRIPTION = """\
agnews\
"""

DIR = {
    "train": "./test.csv",
}

class arc_sup(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            description = _DESCRIPTION,
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
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
        ]

    def _generate_examples(self, filepath):
        logger.info("generating examples from = %s", filepath)
        key = 0
        data = pd.read_csv(filepath)
        for _,row in data.iterrows():
                text = row["Description"]
                label = row["Class Index"]-1
                yield key, {
                    "text": text,
                    "label": label,
                }
                key += 1



if __name__ == "__main__":
    mydata = load_dataset('./ag_news.py')
    print(mydata)
    print(mydata['train'][0])