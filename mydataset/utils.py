import datasets
from datasets import load_dataset, load_dataset_builder, interleave_datasets, concatenate_datasets, load_from_disk
from setfit import sample_dataset

def few_shot(dataset, few_shot=8):
    train_dataset = sample_dataset(dataset["train"], label_column="label", num_samples=few_shot)
    dataset["train"]=train_dataset
    return dataset


def get_restaurant(fs=-1, bias=0):
    def reform(example):
        example["text"] = example["term"] + "[SEP]" + example["text"]
        example["label"] += bias
        return example

    dataset = load_dataset("/mnt/hw/nlp/assignment3/data/SemEval14-res")
    if fs > -1:
        dataset = few_shot(dataset, fs)

    dataset = dataset.map(reform)
    dataset = dataset.remove_columns(["term"])
    num_class = max(dataset["train"]["label"]) + 1
    return dataset, num_class



def get_laptop(fs=-1, bias=0):
    def reform(example):
        example["text"] = example["term"] + "[SEP]" + example["text"]
        example["label"] += bias
        return example

    dataset = load_dataset("/mnt/hw/nlp/assignment3/data/SemEval14-laptop")
    if fs > -1:
        dataset = few_shot(dataset, fs)

    dataset = dataset.map(reform)
    dataset = dataset.remove_columns(["term"])
    num_class = max(dataset["train"]["label"]) + 1
    return dataset, num_class



def get_aclarc(fs=-1, bias=0):
    def reform(example):
        example["label"] += bias
        return example

    dataset = load_dataset("/mnt/hw/nlp/assignment3/data/citation_intent")
    if fs > -1:
        dataset = few_shot(dataset, fs)

    dataset = dataset.map(reform)
    num_class = max(dataset["train"]["label"]) + 1
    return dataset, num_class



def get_agnews(split=True, split_ratio=0.1, fs=-1, bias=0):
    def reform(example):
        example["label"] += bias
        return example

    dataset = load_dataset("/mnt/hw/nlp/assignment3/data/ag_news")
    if split:
        dataset = dataset["train"]
        dataset = dataset.train_test_split(test_size=split_ratio, seed=2022)

    if fs > -1:
        dataset = few_shot(dataset, fs)

    dataset = dataset.map(reform)
    num_class = max(dataset["train"]["label"]) + 1
    return dataset, num_class

# def get_restaurant(tokenizer, tokenize=True, fs=-1, bias=0, max_length=64):
#     def reform(example):
#         example["text"] = example["term"] + tokenizer.sep_token + example["text"]
#         example["label"] += bias
#         if tokenize:
#             example["text"] = tokenizer(example["text"], padding="max_length", max_length=max_length)
#         return example
#
#     dataset = load_dataset("../data/SemEval14-res")
#     if fs > -1:
#         dataset = few_shot(dataset, fs)
#
#     dataset = dataset.map(reform)
#     dataset = dataset.remove_columns(["term"])
#     num_class = max(dataset["train"]["label"]) + 1
#     return dataset, num_class
#
#
#
# def get_laptop(tokenizer, tokenize=True, fs=-1, bias=0, max_length=64):
#     def reform(example):
#         example["text"] = example["term"] + tokenizer.sep_token + example["text"]
#         example["label"] += bias
#         if tokenize:
#             example["text"] = tokenizer(example["text"], padding="max_length", max_length=max_length)
#         return example
#
#     dataset = load_dataset("../data/SemEval14-laptop")
#     if fs > -1:
#         dataset = few_shot(dataset, fs)
#
#     dataset = dataset.map(reform)
#     dataset = dataset.remove_columns(["term"])
#     num_class = max(dataset["train"]["label"]) + 1
#     return dataset, num_class
#
#
#
# def get_aclarc(tokenizer, tokenize=True, fs=-1, bias=0, max_length = 64):
#     def reform(example):
#         example["label"] += bias
#         if tokenize:
#             example["text"] = tokenizer(example["text"], padding="max_length", max_length=max_length)
#         return example
#
#     dataset = load_dataset("../data/citation_intent")
#     if fs > -1:
#         dataset = few_shot(dataset, fs)
#
#     dataset = dataset.map(reform)
#     num_class = max(dataset["train"]["label"]) + 1
#     return dataset, num_class
#
#
#
# def get_agnews(tokenizer, split=True, split_ratio=0.1, tokenize=True, fs=-1, bias=0, max_length=64):
#     def reform(example):
#         example["label"] += bias
#         if tokenize:
#             example["text"] = tokenizer(example["text"], padding="max_length", max_length=max_lenth)
#         return example
#
#     dataset = load_dataset("../data/ag_news")
#     if split:
#         dataset = dataset["train"]
#         dataset = dataset.train_test_split(test_size=split_ratio, seed=2022)
#
#     if fs > -1:
#         dataset = few_shot(dataset, fs)
#
#     dataset = dataset.map(reform)
#     num_class = max(dataset["train"]["label"]) + 1
#     return dataset, num_class



def merge(mydata1, mydata2):
    combined_dataset_train = concatenate_datasets([mydata1["train"], mydata2["train"]], axis=0)
    combined_dataset_test = concatenate_datasets([mydata1["test"], mydata2["test"]], axis=0)
    combined_dataset = datasets.DatasetDict({"train": combined_dataset_train, "test":combined_dataset_test})
    return combined_dataset




if __name__ == "__main__":
    # from transformers import BertTokenizer
    # tokenizer = BertTokenizer.from_pretrained('./bert_tokenizer')

    mydata1, num_class1 = get_restaurant(bias=0)
    mydata2, num_class2 = get_laptop(bias=num_class1)
    mydata3, _ = get_agnews(bias=num_class1+num_class2)
    # print(mydata1["train"]["text"])
    # print(mydata1["train"]["label"])
    print(mydata1)
    print(mydata2)
    print(mydata3)
    combined_dataset = merge(mydata1,mydata2)
    combined_dataset = merge(combined_dataset, mydata3)
    print(combined_dataset)
    # print(datainfo.info.features)
    print(combined_dataset['train'][0])
    print(combined_dataset['train'][4000])
    print(combined_dataset['train'][8000])

    print(combined_dataset['test'][0])
    print(combined_dataset["train"].column_names)
    # print(num_class)
