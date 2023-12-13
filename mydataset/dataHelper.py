from .utils import get_laptop, get_agnews, get_aclarc, get_restaurant, merge

datanam2func = {
    "laptop_sup": get_laptop,
    "laptop_fs": get_laptop,
    "restaurant_sup": get_restaurant,
    "restaurant_fs": get_restaurant,
    "acl_sup": get_aclarc,
    "acl_fs": get_aclarc,
    "agnews_sup": get_agnews,
    "agnews_fs": get_agnews,
}

def get_dataset(data_name_list, few_shot=-1, split=True, split_ratio=0.1):  #“res, lap, acl”
    dnl = data_name_list.lower().replace(" ", "").split(",")    #["res","lap","acl"]
    print(dnl)
    bias = 0
    for idx, data_name in enumerate(dnl):
        func = datanam2func[data_name]
        if data_name in ["agnews_sup", "agnews_fs"]:
            dataset, bias = func(fs=few_shot, bias=bias, split=split, split_ratio=split_ratio)
        else:
            dataset, bias = func(fs=few_shot, bias=bias)

        if idx == 0:
            dataset_out = dataset
        else:
            dataset_out = merge(dataset_out, dataset)
        dataset_out["validation"] = dataset_out["test"]

    return dataset_out, bias


if __name__ == "__main__":
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('/mnt/hw/nlp/assignment3/mydatasets/bert_tokenizer')
    dataset, label_num = get_dataset("restaurant_sup, agnews_sup, acl_sup")
    print(dataset['train'][0])
    print(dataset['train'][4000])
    print(dataset['train'][7000])
    print(dataset)
    print(label_num)